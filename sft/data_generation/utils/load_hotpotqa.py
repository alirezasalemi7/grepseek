"""Load the HotpotQA training set from the FlashRAG_datasets HF repo.

Source: https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets

Returns a list of dicts with the fields we care about:
    {"id": str, "question": str, "golden_answers": list[str],
     "level": str|None, "type": str|None}

`level` is HotpotQA's difficulty label ("easy"|"medium"|"hard") and `type` is
its question type ("bridge"|"comparison"), both pulled out of the dataset's
`metadata` column. Consumers that don't need them (e.g. create_data.py) simply
ignore the extra keys. The bulky rest of `metadata` (context, supporting_facts)
is dropped.
"""
import os
from typing import Optional

from datasets import load_dataset


REPO_ID = "RUC-NLPIR/FlashRAG_datasets"
DEFAULT_CACHE_DIR = os.environ.get("HF_DATASETS_CACHE")  # None -> use HuggingFace default cache


def load_hotpotqa(
    split: str = "train",
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    limit: Optional[int] = None,
) -> list:
    """Load the HotpotQA split from FlashRAG_datasets.

    Args:
        split: "train" or "dev".
        cache_dir: HF cache directory. Defaults to the workspace `.cache`.
        limit: if set, return only the first N examples (useful for debugging).

    Returns:
        A list of dicts: {"id", "question", "golden_answers", "level", "type"}.
    """
    if split not in ("train", "dev"):
        raise ValueError(f"split must be 'train' or 'dev', got {split!r}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    ds = load_dataset(
        REPO_ID,
        data_files={split: f"hotpotqa/{split}.jsonl"},
        split=split,
        cache_dir=cache_dir,
    )

    # Keep id/question/golden_answers plus metadata (we extract level/type from
    # it below, then drop the bulky context/supporting_facts).
    keep = ("id", "question", "golden_answers", "metadata")
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    out = []
    for row in ds.to_list():
        md = row.get("metadata") or {}
        out.append({
            "id": row.get("id"),
            "question": row.get("question"),
            "golden_answers": row.get("golden_answers"),
            "level": md.get("level"),   # "easy" | "medium" | "hard" | None
            "type": md.get("type"),     # "bridge" | "comparison" | None
        })
    return out


if __name__ == "__main__":
    import json

    data = load_hotpotqa(split="train", limit=3)
    print(f"loaded {len(data)} examples (limited preview)")
    print(json.dumps(data, indent=2)[:1000])
