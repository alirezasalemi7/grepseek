"""Pipeline parser + classifier.

Splits a shell command at `|` into stages, identifies how each stage
interacts with state, and decides one of five execution strategies:

    CONCAT     — every stage is stateless (rg/grep/cut/tr/sed -E without
                 in-place / line numbering); run pipeline on each shard,
                 concat outputs in shard order.
    HEAD       — CONCAT shape ending in `head -n N`; apply `head -n N`
                 per shard (bounds memory), concat, apply `head -n N` again.
    COUNT      — CONCAT shape ending in `wc -l`; sum per-shard counts.
    SORT_HEAD  — CONCAT ending in `sort [| uniq] | head -n N`; per-shard
                 sort | head, then sort -m | [uniq |] head -n N.
    FALLBACK   — anything else; engine runs against the single corpus file
                 instead, byte-identical to today.

The classifier is intentionally conservative — every "maybe safe" case routes
to FALLBACK. Correctness > coverage.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Programs we know how to fan out without changing output bytes.
# Critical: each is line-oriented and stateless across shard boundaries when
# combined with a corpus that is split on line boundaries.
_STATELESS_PROGRAMS = {"rg", "grep", "cut", "tr", "sed"}

# rg/grep flags that BREAK shard-independence — flags that emit line numbers,
# filenames with counts, or otherwise produce output that depends on global
# context. If any of these appear we bail to FALLBACK.
_UNSAFE_RG_FLAGS = {
    "-n", "--line-number",      # line numbers would conflict across shards
    "-c", "--count",            # would need summing across shards (different shape than wc -l)
    "-C", "--context",          # context lines across shard boundaries are wrong
    "-A", "--after-context",
    "-B", "--before-context",
    "--json",                   # JSON output has metadata we'd have to merge
    "-l", "--files-with-matches",  # filename-level dedup
    "-L", "--files-without-match",
    "-r", "--recursive",        # we already pass an explicit file
    "--no-filename",            # changes vs default in multi-file context
    "-H", "--with-filename",    # ditto
    "--stats",                  # global stats
}

# Flags after `sed` we treat as unsafe (in-place / file modification).
_UNSAFE_SED_FLAGS = {"-i", "--in-place", "-f"}

# Pattern for `head -n N` or `head -N` as a tail stage of the pipeline.
_HEAD_RE = re.compile(r"^head\s+(?:-n\s*(\d+)|-(\d+))$")
# Pattern for `wc -l` (and only -l; `wc` alone returns chars, lines, bytes; we
# only safely shard the lines-only case).
_WC_L_RE = re.compile(r"^wc\s+-l$")
# `sort` with no args, or with safe flags only (no -u, no -k, no -t etc that
# would change merge semantics non-trivially).
_SORT_PLAIN_RE = re.compile(r"^sort(\s+-[rR])?$")  # allow -r / -R only
# `uniq` with no args (uniq -c would need summing; -u/-d need cross-shard view).
_UNIQ_PLAIN_RE = re.compile(r"^uniq$")


class Strategy(Enum):
    CONCAT = "concat"
    HEAD = "head"
    COUNT = "count"
    SORT_HEAD = "sort_head"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class Stage:
    """One pipeline stage (a single program invocation + args).

    `raw` is the original whitespace-trimmed text for that stage so we can
    re-emit pipelines without re-quoting (which is easy to get wrong).
    """
    raw: str
    prog: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    """Output of classify()."""
    strategy: Strategy
    stages: tuple[Stage, ...]
    # Strategy-specific metadata. For HEAD: the requested N. For SORT_HEAD:
    # the requested N and whether `uniq` is in the pipeline. For COUNT/CONCAT:
    # not used.
    head_n: Optional[int] = None
    has_uniq: bool = False
    # If FALLBACK, the reason — useful for logging/debugging.
    fallback_reason: str = ""


def parse_pipeline(cmd: str) -> list[Stage]:
    """Split `cmd` on `|` and tokenize each stage. Raises ValueError on parse
    failure (mirrors the existing inference.tools validator behavior)."""
    stages: list[Stage] = []
    for seg in (s.strip() for s in cmd.split("|")):
        if not seg:
            raise ValueError("empty pipeline segment")
        try:
            tokens = shlex.split(seg)
        except ValueError as exc:
            raise ValueError(f"could not parse segment {seg!r}: {exc}")
        if not tokens:
            raise ValueError("empty command after tokenizing")
        prog = tokens[0].split("/")[-1]
        stages.append(Stage(raw=seg, prog=prog, tokens=tuple(tokens)))
    return stages


def _stage_is_stateless(stage: Stage) -> tuple[bool, str]:
    """True iff stage produces the same per-shard output bytes as if it had
    seen only that shard's input (i.e. no cross-line / cross-shard state).

    Returns (ok, reason_if_not).
    """
    if stage.prog not in _STATELESS_PROGRAMS:
        return False, f"program {stage.prog!r} not stateless"

    if stage.prog in ("rg", "grep"):
        for tok in stage.tokens[1:]:
            # Bare `-l`, `--count` etc. — bailout list above.
            if tok in _UNSAFE_RG_FLAGS:
                return False, f"unsafe flag {tok!r} on {stage.prog}"
            # Combined short flags like `-rn`, `-In` — be paranoid and
            # bail if any unsafe single-letter flag appears.
            if tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
                for ch in tok[1:]:
                    if f"-{ch}" in _UNSAFE_RG_FLAGS:
                        return False, f"unsafe flag '-{ch}' inside {tok!r}"
        return True, ""

    if stage.prog == "sed":
        for tok in stage.tokens[1:]:
            if tok in _UNSAFE_SED_FLAGS:
                return False, f"unsafe flag {tok!r} on sed"
        return True, ""

    # cut, tr are line-local by definition — always safe.
    return True, ""


def classify(stages: list[Stage]) -> Plan:
    """Decide which strategy to apply to `stages`.

    The classification is shape-based: we look at all stages except possibly
    a special tail (head / wc -l / sort [| uniq] | head). The non-tail stages
    must all be stateless. If the tail matches a known shape, we pick the
    specialized strategy; otherwise CONCAT; otherwise FALLBACK.
    """
    if not stages:
        return Plan(strategy=Strategy.FALLBACK, stages=tuple(), fallback_reason="empty pipeline")

    # First, ensure the head of the pipeline is at least one stateless stage
    # that reads from the corpus (rg/grep/cut/tr/sed). Otherwise the engine
    # doesn't gain anything and we'd risk weird corner cases (e.g. `cat`
    # reading from corpus.jsonl is technically stateless but pre-shard
    # routing is special — bail to fallback).
    head = stages[0]
    if head.prog not in ("rg", "grep"):
        # We require the FIRST stage to actually read from the corpus file.
        # Everything else is downstream filtering and we handle uniformly via
        # the classifier below.
        return Plan(
            strategy=Strategy.FALLBACK, stages=tuple(stages),
            fallback_reason=f"first stage is {head.prog!r}, not rg/grep",
        )

    # Detect special tails.
    last = stages[-1]
    body = stages[:-1]

    # head -n N tail (HEAD or SORT_HEAD)
    head_m = _HEAD_RE.match(last.raw)
    if head_m is not None:
        n_str = head_m.group(1) or head_m.group(2)
        try:
            head_n = int(n_str)
        except ValueError:
            return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                        fallback_reason=f"bad head N: {n_str!r}")
        if head_n <= 0:
            return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                        fallback_reason=f"non-positive head N: {head_n}")

        # SORT_HEAD: body ends in `sort` or `sort | uniq`.
        if body and _SORT_PLAIN_RE.match(body[-1].raw):
            sort_body = body[:-1]
            for s in sort_body:
                ok, reason = _stage_is_stateless(s)
                if not ok:
                    return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                                fallback_reason=f"non-stateless pre-sort: {reason}")
            return Plan(strategy=Strategy.SORT_HEAD, stages=tuple(stages),
                        head_n=head_n, has_uniq=False)
        if len(body) >= 2 and _UNIQ_PLAIN_RE.match(body[-1].raw) and _SORT_PLAIN_RE.match(body[-2].raw):
            sort_body = body[:-2]
            for s in sort_body:
                ok, reason = _stage_is_stateless(s)
                if not ok:
                    return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                                fallback_reason=f"non-stateless pre-sort: {reason}")
            return Plan(strategy=Strategy.SORT_HEAD, stages=tuple(stages),
                        head_n=head_n, has_uniq=True)

        # Plain HEAD: every body stage must be stateless.
        for s in body:
            ok, reason = _stage_is_stateless(s)
            if not ok:
                return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                            fallback_reason=reason)
        return Plan(strategy=Strategy.HEAD, stages=tuple(stages), head_n=head_n)

    # wc -l tail (COUNT)
    if _WC_L_RE.match(last.raw):
        for s in body:
            ok, reason = _stage_is_stateless(s)
            if not ok:
                return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                            fallback_reason=reason)
        return Plan(strategy=Strategy.COUNT, stages=tuple(stages))

    # No special tail — CONCAT only if every stage is stateless.
    for s in stages:
        ok, reason = _stage_is_stateless(s)
        if not ok:
            return Plan(strategy=Strategy.FALLBACK, stages=tuple(stages),
                        fallback_reason=reason)
    return Plan(strategy=Strategy.CONCAT, stages=tuple(stages))
