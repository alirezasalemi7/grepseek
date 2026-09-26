"""Build the RL train + dev JSONL from NQ + HotpotQA (the FlashRAG_datasets repo).

Output schema is exactly what the RL dataset loader (grepseek/data/qa.py
`load_qa_examples`, used by GrepSeekDataset) expects — one JSON object per line:
    {"id": str, "question": str, "golden_answers": list[str], "source": str}
(`source` is informational; the trainer ignores extra keys.)

The train set mixes the two sources and shuffles (seeded). The dev set is a
source-balanced subset of each dev split for cheap eval. Empty-answer rows are
dropped; ids are prefixed `nq_` / `hp_` so they stay unambiguous after mixing.

Usage:
    python prepare_rl_data.py --out_dir data/rl/nq_hotpot          # full NQ+HotpotQA
    python prepare_rl_data.py --out_dir data/rl/small \
        --nq_train_count 5000 --hotpot_train_count 5000 --val_count_per_source 250

Then point the launcher at the output:
    export GREPSEEK_TRAIN_FILES=data/rl/nq_hotpot/train.jsonl
    export GREPSEEK_VAL_FILES=data/rl/nq_hotpot/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset


_REPO = "RUC-NLPIR/FlashRAG_datasets"

_SOURCES = {
    "nq": "nq",
    "hotpotqa": "hotpotqa",
}
_ID_PREFIX = {
    "nq": "nq_",
    "hotpotqa": "hp_",
}


def _load_split(source: str, split: str, cache_dir: Optional[str]) -> list[dict]:
    subfolder = _SOURCES[source]
    ds = load_dataset(
        _REPO,
        data_files={split: f"{subfolder}/{split}.jsonl"},
        split=split,
        cache_dir=cache_dir,
    )
    keep = ("id", "question", "golden_answers")
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds.to_list()


def _normalize_rows(raw_rows: Iterable[dict], source: str) -> list[dict]:
    """Filter empty-answer rows, prefix ids, attach `source` tag."""
    prefix = _ID_PREFIX[source]
    out: list[dict] = []
    for i, row in enumerate(raw_rows):
        question = (row.get("question") or "").strip()
        if not question:
            continue
        golds = row.get("golden_answers") or []
        if isinstance(golds, str):
            golds = [golds]
        golds = [str(g).strip() for g in golds if str(g).strip()]
        if not golds:
            continue

        raw_id = str(row.get("id") or i)
        ex_id = raw_id if raw_id.startswith(prefix) else f"{prefix}{raw_id}"
        out.append({
            "id": ex_id,
            "question": question,
            "golden_answers": golds,
            "source": source,
        })
    return out


def _sample(rows: list[dict], count: Optional[int], rng: random.Random) -> list[dict]:
    """Return `count` rows (without replacement) or all if count is None/<=0/>=len."""
    if count is None or count <= 0 or count >= len(rows):
        return list(rows)
    return rng.sample(rows, count)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _parse_count(arg) -> Optional[int]:
    """'all'/-1/empty -> None (keep all). Int strings -> int."""
    s = str(arg or "").strip().lower()
    if s in ("", "all", "-1"):
        return None
    return int(s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True,
                   help="Output directory; train.jsonl + dev.jsonl land here.")
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"),
                   help="HuggingFace cache dir (defaults to $HF_HOME, else HF's default).")
    p.add_argument("--nq_train_count", default="all",
                   help="NQ train rows to keep, or 'all' (default).")
    p.add_argument("--hotpot_train_count", default="all",
                   help="HotpotQA train rows to keep, or 'all' (default).")
    p.add_argument("--val_count_per_source", default="all",
                   help="Rows per source pulled from each dev split into the val set. "
                        "'all' (default) keeps every dev row; 'none'/0 skips dev.jsonl; "
                        "an int subsamples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_shuffle", action="store_true",
                   help="Skip the final shuffle (keeps per-source ordering).")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"seed": args.seed, "splits": {}}
    nq_target = _parse_count(args.nq_train_count)
    hp_target = _parse_count(args.hotpot_train_count)

    train_rows: list[dict] = []
    for source, want in (("nq", nq_target), ("hotpotqa", hp_target)):
        raw = _load_split(source, "train", args.cache_dir)
        normalized = _normalize_rows(raw, source)
        sampled = _sample(normalized, want, rng)
        train_rows.extend(sampled)
        summary["splits"].setdefault("train", {})[source] = {
            "loaded": len(raw), "after_filter": len(normalized), "kept": len(sampled),
        }
        print(f"[train] {source}: loaded={len(raw)}  after_filter={len(normalized)}  "
              f"kept={len(sampled)}", flush=True)

    if not args.no_shuffle:
        rng.shuffle(train_rows)
    train_path = out_dir / "train.jsonl"
    _write_jsonl(train_rows, train_path)
    summary["splits"]["train"]["total"] = len(train_rows)
    summary["splits"]["train"]["path"] = str(train_path)
    print(f"[train] wrote {len(train_rows)} rows -> {train_path}", flush=True)

    val_arg = str(args.val_count_per_source).strip().lower()
    skip_val = val_arg in ("none", "0", "skip")
    val_target = None if skip_val else _parse_count(args.val_count_per_source)
    if not skip_val:
        val_rows: list[dict] = []
        for source in ("nq", "hotpotqa"):
            raw = _load_split(source, "dev", args.cache_dir)
            normalized = _normalize_rows(raw, source)
            sampled = _sample(normalized, val_target, rng)
            val_rows.extend(sampled)
            summary["splits"].setdefault("dev", {})[source] = {
                "loaded": len(raw), "after_filter": len(normalized), "kept": len(sampled),
            }
            print(f"[dev]   {source}: loaded={len(raw)}  after_filter={len(normalized)}  "
                  f"kept={len(sampled)}", flush=True)
        if not args.no_shuffle:
            rng.shuffle(val_rows)
        dev_path = out_dir / "dev.jsonl"
        _write_jsonl(val_rows, dev_path)
        summary["splits"]["dev"]["total"] = len(val_rows)
        summary["splits"]["dev"]["path"] = str(dev_path)
        print(f"[dev]   wrote {len(val_rows)} rows -> {dev_path}", flush=True)

    summary_path = out_dir / "build_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
