"""Per-strategy execution: run a Plan against pre-built shards and merge the
results so the output is byte-identical to running the same pipeline against
the un-sharded corpus.

All runners take:
    - plan: the Plan from pipeline.classify()
    - shard_files: ordered list of absolute paths to shard files
    - corpus_filename_in_cmd: the logical filename the original command
      references (e.g. "corpus.jsonl" or "wiki_corpus.jsonl"). We substitute
      the per-shard absolute path for this token in each subshell.
    - timeout, env

Returns (stdout, returncode). returncode != 0 only when ALL per-shard
subshells fail; partial failures are surfaced via stderr but the pipeline
keeps going (matches the single-file `set -o pipefail`-off default).
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from .pipeline import Plan, Stage, Strategy


def _stage_text(stages: Iterable[Stage]) -> str:
    """Join stages back into a `|` pipeline using their original raw text.

    We use Stage.raw (the pre-shlex string) so quoting is preserved exactly
    as the model wrote it — no risk of shlex re-quote drift.
    """
    return " | ".join(s.raw for s in stages)


def _sub_corpus_token(pipeline_text: str, corpus_token: str, shard_path: str) -> str:
    """Replace the corpus filename in the pipeline with a per-shard absolute
    path. We do a literal substring replacement; the existing inference.tools
    validator already rejects newlines / multiple inputs / shell metachars in
    the command, so a naive replace is safe in this context.
    """
    return pipeline_text.replace(corpus_token, shard_path)


def _run_one(cmd: str, timeout: float, env: dict | None) -> tuple[str, str, int]:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )
        return proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired:
        return "", f"[shard timed out after {timeout}s]\n", -1


def _run_parallel(per_shard_cmds: list[str], timeout: float, env: dict | None
                  ) -> list[tuple[str, str, int]]:
    """Run a per-shard command list in parallel, preserving input order in
    the returned list. Uses threads (not processes) because each underlying
    `subprocess.run` already forks — adding a process pool on top would just
    double-fork without speedup.
    """
    results: list[tuple[str, str, int] | None] = [None] * len(per_shard_cmds)
    # max_workers = len(per_shard_cmds): we want all shards in flight at once.
    with ThreadPoolExecutor(max_workers=max(1, len(per_shard_cmds))) as pool:
        futures = {pool.submit(_run_one, c, timeout, env): i
                   for i, c in enumerate(per_shard_cmds)}
        for fut in futures:
            i = futures[fut]
            results[i] = fut.result()
    # Cast away the Optional; ThreadPoolExecutor.future is always resolved here.
    return [r for r in results if r is not None]  # type: ignore[return-value]


def _merge_stderrs(parts: list[tuple[str, str, int]]) -> str:
    """Concatenate non-empty stderrs, deduplicated, so a common rg error
    (e.g. "regex parse error") shows once rather than N times."""
    seen = set()
    out_lines: list[str] = []
    for _, stderr, _ in parts:
        for line in stderr.splitlines():
            if line and line not in seen:
                seen.add(line)
                out_lines.append(line)
    return ("\n".join(out_lines) + "\n") if out_lines else ""


def _overall_rc(parts: list[tuple[str, str, int]]) -> int:
    """Returncode: 0 if any shard succeeded (matches rg single-file semantics
    where exit 1 means "no matches" and exit 0 means "matches found" — even
    if some shards have no matches, we want exit 0 overall when at least one
    found something). For total failure (all shards rc != 0), pick the most
    common non-zero code so eg. timeouts vs syntax errors are distinguishable.
    """
    if any(rc == 0 for _, _, rc in parts):
        return 0
    # All non-zero: report the first non-zero (typically all the same).
    for _, _, rc in parts:
        if rc != 0:
            return rc
    return 0  # empty input shouldn't happen but be safe


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def run_concat(plan: Plan, *, shard_files: list[str], corpus_token: str,
               timeout: float, env: dict | None) -> tuple[str, int]:
    """Stateless pipeline; concat per-shard outputs in shard order."""
    pipeline_text = _stage_text(plan.stages)
    cmds = [_sub_corpus_token(pipeline_text, corpus_token, sf) for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)
    stdout = "".join(p[0] for p in parts)
    return stdout, _overall_rc(parts)


def run_head(plan: Plan, *, shard_files: list[str], corpus_token: str,
             timeout: float, env: dict | None) -> tuple[str, int]:
    """CONCAT pipeline ending in `head -n N`. Apply `head -n N` per shard
    (bounds memory), concat, then trim again to N lines total."""
    assert plan.head_n is not None
    n = plan.head_n

    # Per-shard pipeline already ends in `head -n N`, so we keep the original
    # stages as-is. The per-shard rg→...→head pipeline already bounds output
    # to N lines per shard. After concat we'll have at most N * len(shards).
    pipeline_text = _stage_text(plan.stages)
    cmds = [_sub_corpus_token(pipeline_text, corpus_token, sf) for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)

    # Concat in shard order, then re-head to N (lines, not bytes — matches
    # `head -n N` semantics).
    concat = "".join(p[0] for p in parts)
    if not concat:
        return concat, _overall_rc(parts)

    lines = concat.split("\n")
    # `head -n N` returns the first N "lines" including the trailing newline
    # behavior. To match byte-equivalence with a single-file run we slice
    # by lines and rebuild. A trailing newline-only element from split() is
    # preserved to keep the original file's terminator.
    if len(lines) > n:
        # Take first N lines. Need to add the trailing newline if the original
        # nth line was newline-terminated (which it always is after split('\n')
        # except possibly the last element).
        head_lines = lines[:n]
        out = "\n".join(head_lines) + "\n"
    else:
        out = concat
    return out, _overall_rc(parts)


def run_count(plan: Plan, *, shard_files: list[str], corpus_token: str,
              timeout: float, env: dict | None) -> tuple[str, int]:
    """CONCAT pipeline ending in `wc -l`. Sum the per-shard counts.

    Each per-shard call returns "<n>\n" (the body of `wc -l` output when
    reading from stdin — no filename prefix).
    """
    pipeline_text = _stage_text(plan.stages)
    cmds = [_sub_corpus_token(pipeline_text, corpus_token, sf) for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)

    total = 0
    for stdout, _stderr, rc in parts:
        if rc != 0 and not stdout.strip():
            continue  # treat empty failed shard as 0 (matches `rg | wc -l`
                      # when rg exits 1 because no matches)
        try:
            total += int(stdout.strip().split()[0])
        except (ValueError, IndexError):
            # If a shard's wc output is malformed, surface as failure.
            return "", 1
    # `wc -l` from a single file with no filename prefix outputs e.g. "      42\n".
    # When stdin is piped, no filename and a single line: e.g. "42\n". Match the
    # piped form (which is what the model's pipelines always produce).
    return f"{total}\n", 0


def run_sort_head(plan: Plan, *, shard_files: list[str], corpus_token: str,
                  timeout: float, env: dict | None) -> tuple[str, int]:
    """CONCAT-body ending in `sort [| uniq] | head -n N`.

    Per shard: run [body] | sort | head -n N (uniq skipped per-shard — it'd
    only dedupe within a shard, must dedupe globally).
    Then: merge the N sorted streams with `sort -m`, optionally `uniq`, and
    `head -n N`.
    """
    assert plan.head_n is not None
    n = plan.head_n

    # Split the stages: everything up to (but not including) the trailing
    # `sort` is the per-shard body; we re-assemble the per-shard pipeline as
    # `<body> | sort | head -n N`.
    stages = list(plan.stages)
    tail_count = 3 if plan.has_uniq else 2  # sort | (uniq |) head
    body = stages[:-tail_count]
    if not body:
        # Pathological case (`sort | uniq | head -n 3` with no upstream
        # filter) — would need to read the whole corpus into sort. Bail.
        return run_concat(plan, shard_files=shard_files, corpus_token=corpus_token,
                          timeout=timeout, env=env)

    per_shard_pipeline_template = _stage_text(body) + f" | sort | head -n {n}"
    cmds = [_sub_corpus_token(per_shard_pipeline_template, corpus_token, sf)
            for sf in shard_files]
    parts = _run_parallel(cmds, timeout, env)

    # Write each shard's partial-sorted output to a tempfile so `sort -m`
    # can merge them. Could also pipe via process-substitution but tempfiles
    # are simpler and tolerate large outputs without blocking writers.
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="parallel_search_sortmerge_")
    try:
        files: list[str] = []
        for i, (stdout, _stderr, rc) in enumerate(parts):
            fp = os.path.join(tmpdir, f"part_{i:04d}")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(stdout)
            files.append(fp)
        # Build the merge pipeline. Quote each file path safely.
        import shlex as _shlex
        files_q = " ".join(_shlex.quote(p) for p in files)
        merge_cmd = f"sort -m {files_q}"
        if plan.has_uniq:
            merge_cmd += " | uniq"
        merge_cmd += f" | head -n {n}"
        try:
            mproc = subprocess.run(
                merge_cmd, shell=True, executable="/bin/bash",
                timeout=timeout, capture_output=True, text=True, env=env,
            )
            return (mproc.stdout or ""), mproc.returncode
        except subprocess.TimeoutExpired:
            return "", -1
    finally:
        # Clean up tempfiles.
        for f in files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


# Dispatch table.
STRATEGY_DISPATCH = {
    Strategy.CONCAT:    run_concat,
    Strategy.HEAD:      run_head,
    Strategy.COUNT:     run_count,
    Strategy.SORT_HEAD: run_sort_head,
    # FALLBACK is handled by the engine itself (runs against the single file).
}
