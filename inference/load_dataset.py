"""Loaders for the Search-R1 benchmark suite (FlashRAG datasets).

Search-R1 reports on seven QA datasets, all of which are mirrored in
RUC-NLPIR's `FlashRAG_datasets` repository on HuggingFace. Each dataset has the
same JSON schema:

    {"id": str, "question": str, "golden_answers": [str, ...], "metadata": ...}

`load_flashrag_dataset` normalizes that into a uniform list of dicts
`{"id": str, "question": str, "golden_answers": list[str]}` so the rest of the
inference pipeline (agent, scorer) does not need to know which dataset it is
running on.

Single-hop:  nq, triviaqa, popqa
Multi-hop:   hotpotqa, 2wikimultihopqa, musique, bamboogle
"""
from __future__ import annotations

import json
import os
from typing import Optional

from datasets import load_dataset as hf_load_dataset


_REPO = "RUC-NLPIR/FlashRAG_datasets"

# Search-R1 canonical evaluation suite. Keys are the names users pass on the
# CLI; values are the subfolder names inside the FlashRAG HF repo.
SEARCH_R1_DATASETS = {
    "nq": "nq",
    "triviaqa": "triviaqa",
    "popqa": "popqa",
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "bamboogle": "bamboogle",
}

# bamboogle only ships a test split in FlashRAG (no train/dev). The rest use
# `dev` or `test` interchangeably depending on the dataset author's convention.
_DEFAULT_SPLIT = {
    "nq": "test",
    "triviaqa": "test",
    "popqa": "test",
    "hotpotqa": "dev",
    "2wikimultihopqa": "dev",
    "musique": "dev",
    "bamboogle": "test",
}


def list_datasets() -> list[str]:
    return sorted(SEARCH_R1_DATASETS.keys())


def default_split(name: str) -> str:
    return _DEFAULT_SPLIT.get(name, "test")


def _row_to_example(row: dict, dataset_name: str) -> Optional[dict]:
    """Normalize one FlashRAG row to {id, question, golden_answers}.

    Returns None for rows that are missing a question or have no answers."""
    question = (row.get("question") or "").strip()
    if not question:
        return None

    golds = row.get("golden_answers") or row.get("answers") or []
    if isinstance(golds, str):
        golds = [golds]
    golds = [str(g).strip() for g in golds if str(g).strip()]
    if not golds:
        return None

    raw_id = row.get("id")
    if raw_id is None:
        raw_id = row.get("_id") or ""
    ex_id = f"{dataset_name}_{raw_id}" if raw_id != "" else f"{dataset_name}_anon"

    return {"id": ex_id, "question": question, "golden_answers": golds}


def load_flashrag_dataset(
    name: str,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    """Load one Search-R1 dataset as a list of `{id, question, golden_answers}`.

    Args:
        name: dataset short name (one of `SEARCH_R1_DATASETS`).
        split: HF split. If None, uses the dataset's default (see `_DEFAULT_SPLIT`).
        cache_dir: HuggingFace cache. Defaults to the workspace-wide cache.
        limit: cap on number of returned rows (after `offset` is applied).
        offset: skip this many rows from the start (useful for sharding).

    Empty-answer rows are filtered out (the scorer needs at least one gold to
    grade against).
    """
    if name not in SEARCH_R1_DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; supported: {sorted(SEARCH_R1_DATASETS)}"
        )

    subfolder = SEARCH_R1_DATASETS[name]
    split = split or default_split(name)
    cache_dir = cache_dir or os.environ.get("HF_HOME")  # None -> HF default cache

    # FlashRAG ships each dataset as `<subfolder>/<split>.jsonl` under a single
    # repo. `data_files=` lets us pick exactly one split without downloading the
    # whole repo.
    ds = hf_load_dataset(
        _REPO,
        data_files={split: f"{subfolder}/{split}.jsonl"},
        split=split,
        cache_dir=cache_dir,
    )

    rows: list[dict] = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if limit is not None and len(rows) >= limit:
            break
        ex = _row_to_example(row, name)
        if ex is None:
            continue
        rows.append(ex)
    return rows


def load_questions_file(
    path: str,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    """Load arbitrary questions from a JSON / JSONL file for **generation** (no
    benchmark, gold answers optional).

    Each record needs at least a question under `question` or `query`; an `id`
    (or `qid`) is used if present, else one is assigned by position. `golden_answers`
    (or `answers`/`answer`) is kept when present so you *can* score later, but it
    is not required — rows without gold are still returned.

    Accepts either a `.jsonl` (one object per line) or a `.json` file holding a
    list of objects (or a `{"data": [...]}` wrapper).
    """
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            records = [json.loads(ln) for ln in f if ln.strip()]
        else:
            payload = json.load(f)
            records = payload.get("data", payload) if isinstance(payload, dict) else payload

    stem = os.path.splitext(os.path.basename(path))[0]
    rows: list[dict] = []
    for i, row in enumerate(records):
        if i < offset:
            continue
        if limit is not None and len(rows) >= limit:
            break
        question = (row.get("question") or row.get("query") or "").strip()
        if not question:
            continue
        ex_id = str(row.get("id") or row.get("qid") or f"{stem}_{i}")
        golds = row.get("golden_answers") or row.get("answers") or row.get("answer") or []
        if isinstance(golds, str):
            golds = [golds]
        golds = [str(g).strip() for g in golds if str(g).strip()]
        rows.append({"id": ex_id, "question": question, "golden_answers": golds})
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
