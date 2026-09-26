"""One-time corpus-shard builder.

Splits `wiki_corpus.jsonl` into N line-aligned shards using `split -d -n l/N`
(part of GNU coreutils — available everywhere ripgrep is). Writes a
manifest.json the engine reads at init.

Idempotent: re-running with the same --src/--dst/--n is a no-op if the
manifest already matches.

Usage:
    python -m inference.parallel_search.sharder \\
        --src /path/to/wiki_18_corpus/wiki_corpus.jsonl \\
        --dst /path/to/shards_16 \\
        --n 16

Validation:
    cat shards_16/shard_* | diff - wiki_corpus.jsonl    # should be empty
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def _sha256(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_lines(path: str) -> int:
    """Count lines via `wc -l` (fast on a 14GB file vs Python iteration)."""
    out = subprocess.check_output(["wc", "-l", path], text=True)
    return int(out.strip().split()[0])


def build_shards(src: str, dst_dir: str, n: int, force: bool = False) -> dict:
    """Build N line-aligned shards from `src` in `dst_dir`. Returns the
    manifest dict written to dst_dir/manifest.json."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"source corpus not found: {src}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    os.makedirs(dst_dir, exist_ok=True)

    manifest_path = os.path.join(dst_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not force:
        with open(manifest_path) as f:
            existing = json.load(f)
        # Sanity: same source SHA and same N → reuse.
        src_sha = _sha256(src)
        if existing.get("src_sha256") == src_sha and existing.get("n_shards") == n:
            all_present = all(os.path.isfile(os.path.join(dst_dir, name))
                              for name in existing.get("shards", []))
            if all_present:
                print(f"[sharder] manifest at {manifest_path} matches "
                      f"(src_sha + n_shards); reusing existing shards.")
                return existing
            print("[sharder] manifest matches but some shards missing; rebuilding.")
        else:
            print(f"[sharder] manifest mismatch (src_sha or n changed); rebuilding.")

    src_sha = _sha256(src)
    total_lines = _count_lines(src)
    print(f"[sharder] src={src}  lines={total_lines:,}  sha={src_sha[:12]}...  n={n}")

    # `split -d -n l/N <src> <prefix>` produces files prefix00, prefix01, ...
    # We use a 4-digit numeric suffix via -a 4 for up to 9999 shards (well
    # beyond what we'd ever use).
    prefix = os.path.join(dst_dir, "shard_")
    # Remove any stale shard files from previous runs with a different N.
    for entry in os.listdir(dst_dir):
        if entry.startswith("shard_"):
            os.unlink(os.path.join(dst_dir, entry))

    width = max(2, len(str(n - 1)))
    cmd = ["split", "-d", "-a", str(width), "-n", f"l/{n}", src, prefix]
    print(f"[sharder] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Collect resulting shard filenames in sorted order.
    shards = sorted(
        entry for entry in os.listdir(dst_dir)
        if entry.startswith("shard_") and not entry.endswith("/")
    )
    if len(shards) != n:
        # split's `-n l/N` should always produce N files when src has > N lines.
        raise RuntimeError(
            f"split produced {len(shards)} shards, expected {n}. "
            f"Is the corpus smaller than {n} lines?"
        )

    # Verify line count round-trips.
    shard_lines = sum(_count_lines(os.path.join(dst_dir, s)) for s in shards)
    if shard_lines != total_lines:
        raise RuntimeError(
            f"line-count mismatch: shards={shard_lines}, src={total_lines}. "
            f"This should not happen with `split -n l/N` — investigate before using."
        )

    manifest = {
        "src": os.path.abspath(src),
        "src_sha256": src_sha,
        "src_lines": total_lines,
        "n_shards": n,
        "shards": shards,                # filenames only, relative to dst_dir
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[sharder] wrote {manifest_path}")
    print(f"[sharder] {n} shards in {dst_dir}, total {total_lines:,} lines")
    return manifest


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="path to wiki_corpus.jsonl")
    p.add_argument("--dst", required=True, help="output directory for shards + manifest")
    p.add_argument("--n", type=int, default=16, help="number of shards (default: 16)")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if manifest matches the source SHA + N")
    args = p.parse_args()
    build_shards(args.src, args.dst, args.n, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
