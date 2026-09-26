"""Offline builder for PACE's auxiliary structure: a singletons-first
vocabulary, Elias-Fano posting lists, a pruned vocabulary suffix array, a
separator-rank bitvector, and blocked line offsets, built once over a flat
JSONL corpus so `inference/tools.py`'s shell-tool executor can narrow
literal `rg -F` / `grep -F` searches to candidate lines instead of scanning
the full corpus file on every call. See `inference/pace/core.py` for the
on-disk layout and its correctness contract.

One-time, offline, single-machine build -- run on a high-memory node. The
corpus is streamed once in binary mode; postings accumulate in memory
before being flushed to disk, so a large corpus needs a correspondingly
large amount of RAM.

Usage:
    python -m inference.pace.build --corpus /path/to/corpus.jsonl --out_dir /path/to/pace_dir
"""
from __future__ import annotations

import argparse

from .core import build_pace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="Path to the flat JSONL corpus")
    parser.add_argument("--out_dir", required=True, help="Output directory for the built PACE structure")
    parser.add_argument("--min-term-len", type=int, default=2, help="Skip terms shorter than this")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar")
    args = parser.parse_args()

    metadata = build_pace(
        args.corpus,
        args.out_dir,
        min_term_len=args.min_term_len,
        progress=not args.no_progress,
    )
    print(f"Built PACE structure at {args.out_dir}")
    print(metadata.to_json())


if __name__ == "__main__":
    main()
