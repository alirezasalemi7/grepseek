"""Load Natural Questions (NQ-Open) from the FlashRAG_datasets HF repo.

Source: https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets

Returns a list of dicts with only the fields we care about:
    {"id": str, "question": str, "golden_answers": list[str]}

NQ-Open is the open-domain (no-context) version of Natural Questions: each
example has a question and a list of valid short-answer surface forms. Suitable
for retrieval-augmented agentic search; the agent must discover the supporting
passage on its own.
"""
import os
from typing import Optional

from datasets import load_dataset


REPO_ID = "RUC-NLPIR/FlashRAG_datasets"
DEFAULT_CACHE_DIR = os.environ.get("HF_DATASETS_CACHE")  # None -> use HuggingFace default cache


def load_nq(
    split: str = "train",
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    limit: Optional[int] = None,
) -> list:
    """Load the NQ-Open split from FlashRAG_datasets.

    Args:
        split: "train" or "dev".
        cache_dir: HF cache directory. Defaults to the workspace `.cache`.
        limit: if set, return only the first N examples (useful for debugging).

    Returns:
        A list of dicts: {"id", "question", "golden_answers"}. Empty-answer
        examples are filtered out (they appear in some NQ exports). Ids are
        prefixed with "nq_" so they don't collide with HotpotQA's
        "train_<n>" ids in cross-run dedup files.
    """
    if split not in ("train", "dev"):
        raise ValueError(f"split must be 'train' or 'dev', got {split!r}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    ds = load_dataset(
        REPO_ID,
        data_files={split: f"nq/{split}.jsonl"},
        split=split,
        cache_dir=cache_dir,
    )

    keep = ("id", "question", "golden_answers")
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows = ds.to_list()
    out = []
    for i, ex in enumerate(rows):
        ans = ex.get("golden_answers") or []
        if not any((a or "").strip() for a in ans):
            continue
        # FlashRAG NQ ids look like "test_<n>" / "train_<n>" — same shape as
        # HotpotQA's. Re-prefix with "nq_" for unambiguous cross-dataset ids.
        raw_id = str(ex.get("id", "") or f"{split}_{i}")
        ex["id"] = raw_id if raw_id.startswith("nq_") else f"nq_{raw_id}"
        out.append(ex)
    return out


if __name__ == "__main__":
    import json

    data = load_nq(split="train", limit=3)
    print(f"loaded {len(data)} examples (limited preview)")
    print(json.dumps(data, indent=2)[:1000])
