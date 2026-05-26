"""Safe shell-tool executor for the GrepSeek inference agent.

Mirrors the executor used by SFT data-gen and RL rollout so the model sees
identical tool behavior at evaluation time. Specifically:

- Whitelist of read-only commands (rg, grep, find, sed, awk, head, tail, cat,
  ls, wc, sort, cut, uniq, tr). No redirection, chaining, or substitution.
- Rewrites `corpus.jsonl` -> `wiki_corpus.jsonl` before execution, so the agent
  can refer to the corpus by the short logical name it was trained on.
- Token-based stdout truncation (default 2048 tokens), matching the SFT
  pipeline's --tool_max_tokens 2048.

Returns the same JSON payload shape the SFT trajectory's tool-role messages
use, so the evaluation trace is comparable to training data byte-for-byte.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = {
    "rg", "grep", "find", "sed", "awk", "head", "tail", "cat",
    "ls", "wc", "sort", "cut", "uniq", "tr",
}

CORPUS_FILENAME_LOGICAL = "corpus.jsonl"
CORPUS_FILENAME_ACTUAL = "wiki_corpus.jsonl"


class ToolError(Exception):
    pass


def _strip_quoted(s: str) -> str:
    s = re.sub(r"'[^']*'", "", s)
    s = re.sub(r'"[^"]*"', "", s)
    return s


_DANGEROUS_PATTERNS = [
    (re.compile(r"\bsed\b[^|]*\s-(?:[a-zA-Z]*i|-in-place)\b"), "sed -i / --in-place"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s-i\s+inplace\b"), "awk -i inplace"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s--in-place\b"), "awk --in-place"),
    (re.compile(r"\bfind\b[^|]*\s-delete\b"), "find -delete"),
    (re.compile(r"\bfind\b[^|]*\s-exec(?:dir)?\b"), "find -exec / -execdir"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bsystem\s*\("), "awk system() call"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bprintf?\b[^|]*>\s*[\"']"), "awk print > file"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\|\s*[\"']"), "awk pipe-to-shell-command"),
    (re.compile(r"\bsed\b[^|]*'\s*e\s+"), "sed 'e' (execute shell) command"),
    (re.compile(r"\b(?:rm|mv|cp|dd|chmod|chown|chgrp|truncate|install|tee|ln)\b"),
     "destructive/write command"),
]


def validate_pipeline(cmd: str) -> None:
    if "\n" in cmd:
        raise ToolError("newlines not allowed in command")
    if len(cmd) > 2000:
        raise ToolError("command too long")
    for pat, label in _DANGEROUS_PATTERNS:
        if pat.search(cmd):
            raise ToolError(f"disallowed: {label}")
    bare = _strip_quoted(cmd)
    for bad in ("`", "$(", "&&", "||", ";", ">", "<"):
        if bad in bare:
            raise ToolError(f"disallowed shell construct: {bad!r}")
    for seg in (s.strip() for s in cmd.split("|")):
        if not seg:
            raise ToolError("empty pipeline segment")
        try:
            tokens = shlex.split(seg)
        except ValueError as e:
            raise ToolError(f"could not parse segment: {e}")
        if not tokens:
            raise ToolError("empty command")
        prog = tokens[0].split("/")[-1]
        if prog not in ALLOWED_COMMANDS:
            raise ToolError(f"command {prog!r} not in whitelist")


# ---------------------------------------------------------------------------
# rg/grep flag injection (Tier 1C — free speedup, no semantic change)
# ---------------------------------------------------------------------------
# `--mmap`     : rg already auto-picks mmap for >MB files on Linux; explicit
#                pins the choice across rg versions / odd filesystems.
# `--no-config`: skip reading user ripgrep config (avoids stray rcfile defaults
#                changing behavior on the cluster).
# `LC_ALL=C`   : bytewise comparison, no locale collation. Safe for our usage
#                (mostly `rg -F` literal patterns); rg's `-i` case-folding is
#                always Unicode-aware regardless of LC_ALL.
_RG_INJECTED_FLAGS = ("--mmap", "--no-config")
_GREP_INJECTED_FLAGS = ("--mmap",)  # GNU grep accepts --mmap but not --no-config


def _augment_rg_flags(cmd: str) -> str:
    """Insert speedup flags after each rg/grep program token.

    Idempotent: if the flag is already present in a stage, it's skipped.
    Operates by `|`-stage so each stage's first token is its program.
    """
    stages = cmd.split("|")
    out_stages: list[str] = []
    for raw in stages:
        seg = raw.strip()
        if not seg:
            out_stages.append(raw)
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            # Unparseable segment — leave alone; the validator (or rg itself)
            # will surface a useful error downstream.
            out_stages.append(raw)
            continue
        if not tokens:
            out_stages.append(raw)
            continue
        prog = tokens[0].split("/")[-1]
        if prog == "rg":
            flags = _RG_INJECTED_FLAGS
        elif prog == "grep":
            flags = _GREP_INJECTED_FLAGS
        else:
            out_stages.append(raw)
            continue
        new_flags = [f for f in flags if f not in tokens]
        if not new_flags:
            out_stages.append(raw)
            continue
        # Splice flags in right after the program name so they precede other
        # args (rg parses flags positionally before non-flags).
        # Use string replace to keep the original whitespace/quoting intact.
        prefix = tokens[0]
        # Find the program token in the raw segment and insert after it.
        idx = seg.find(prefix)
        if idx < 0:
            out_stages.append(raw)
            continue
        insert_at = idx + len(prefix)
        inserted = " " + " ".join(new_flags)
        new_seg = seg[:insert_at] + inserted + seg[insert_at:]
        out_stages.append(new_seg)
    return "|".join(out_stages)


def _build_tool_env() -> dict:
    """Return an env dict with LC_ALL=C overlaid on the parent env. Safe for
    `rg -F`/literal patterns; LC_ALL=C does not change rg's Unicode case
    folding (rg uses an internal table regardless of locale)."""
    return {**os.environ, "LC_ALL": "C"}


# ---------------------------------------------------------------------------
# Tokenizer-based output truncation
# ---------------------------------------------------------------------------

def truncate_tokens(text: str, max_tokens: int, tokenizer) -> tuple[str, bool]:
    """Return (text, was_truncated). Same semantics as
    cold_start_data_gen/utils/pipeline.py:truncate_tokens.

    Tier 3 shortcut: Qwen BPE is >=3 chars/token, so any text with
    len(text) <= max_tokens*3 is guaranteed to fit and we can skip the
    tokenizer call entirely. Saves ~10-50ms per tool response on the
    common `| head -n 3` case where each response is a few KB.
    """
    if not text:
        return text, False
    if len(text) <= max_tokens * 3:
        return text, False
    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # HF fast tokenizer (Rust `tokenizers` lib) can panic on rare byte
        # sequences with PanicException: NormalizedString bad split. CRITICAL:
        # that PanicException subclasses BaseException, NOT Exception, so
        # `except Exception` does NOT catch it and the panic would crash the
        # eval. Catch BaseException (re-raising only interactive/exit signals)
        # and fall back to a conservative char cap.
        char_cap = max_tokens * 3
        if len(text) <= char_cap:
            return text, False
        return text[:char_cap], True
    if len(ids) <= max_tokens:
        return text, False
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=False), True


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    information_lines: list

    def to_payload(self) -> str:
        """JSON shape that matches SFT tool-role messages."""
        return json.dumps(
            {
                "command": self.command,
                "stdout": self.stdout,
                "stderr": self.stderr,
                "exit_code": self.exit_code,
                "timed_out": self.timed_out,
                "truncated": self.truncated,
                "information_lines": self.information_lines,
            },
            ensure_ascii=False,
        )


def run_tool(
    cmd: str,
    *,
    corpus_dir: str,
    timeout: float,
    max_tokens: int,
    tokenizer,
    engine=None,
) -> ToolResult:
    """Validate, rewrite corpus.jsonl -> wiki_corpus.jsonl, execute, truncate.

    If `engine` is provided (a parallel_search.ShardedSearchEngine), it handles
    execution — fanning out parallel-safe pipelines across shards and falling
    back internally to single-file execution for anything it doesn't recognize.
    The engine guarantees byte-equivalent output vs the single-file path.
    When `engine is None` (default), behavior is exactly as before.
    """
    try:
        validate_pipeline(cmd)
    except ToolError as exc:
        return ToolResult(
            command=cmd,
            stdout="",
            stderr=f"validation error: {exc}",
            exit_code=-2,
            timed_out=False,
            truncated=False,
            information_lines=[],
        )

    cmd_to_run = cmd.replace(CORPUS_FILENAME_LOGICAL, CORPUS_FILENAME_ACTUAL)
    cmd_to_run = _augment_rg_flags(cmd_to_run)
    tool_env = _build_tool_env()
    timed_out = False
    stderr = ""
    if engine is not None:
        # Engine handles its own subprocess + per-shard timeout + fallback to
        # the single corpus file. Returncode == -1 indicates a timeout.
        stdout, exit_code = engine.execute(cmd_to_run, env=tool_env)
        if exit_code == -1:
            timed_out = True
            stdout = f"[command timed out after {timeout}s]"
    else:
        try:
            proc = subprocess.run(
                cmd_to_run,
                shell=True,
                executable="/bin/bash",
                cwd=corpus_dir,
                timeout=timeout,
                capture_output=True,
                text=True,
                env=tool_env,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            stdout = f"[command timed out after {timeout}s]"
            stderr = ""
            exit_code = -1
            timed_out = True

    stdout, truncated = truncate_tokens(stdout, max_tokens, tokenizer)
    if truncated:
        stdout += f"\n[... output truncated at {max_tokens} tokens]"

    info_lines = [ln for ln in stdout.split("\n") if ln.strip()]
    return ToolResult(
        command=cmd,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        truncated=truncated,
        information_lines=info_lines,
    )
