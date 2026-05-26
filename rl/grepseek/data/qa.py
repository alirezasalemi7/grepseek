"""Generic QA-example loader for GrepSeek RL training.

Reads a JSONL (or JSON) file of QA records and returns a list of
`DatasetExample`. Dataset-agnostic: any record with `id`/`qid`,
`question`/`query`, and `golden_answers` (FlashRAG-style) works — NQ, HotpotQA,
MuSiQue, etc. The paper's RL trains on NQ + HotpotQA. Answer fields are accepted
as `golden_answers`, `answers`, or `answer` (list or scalar).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qid: str
    query: str
    golden_answers: list[str]
    context_paragraphs: list["ContextParagraph"] = Field(default_factory=list)


class ContextParagraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idx: int
    title: str
    contents: str


def _load_raw_records(path: Path, split: str | None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        if split:
            if records and not any("split" in record for record in records):
                return records
            return [record for record in records if record.get("split") == split]
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if split and split in payload and isinstance(payload[split], list):
            return payload[split]
        if "data" in payload and isinstance(payload["data"], list):
            data = payload["data"]
            if split:
                if data and not any("split" in record for record in data):
                    return data
                return [record for record in data if record.get("split") == split]
            return data
    if isinstance(payload, list):
        if split:
            if payload and not any("split" in record for record in payload if isinstance(record, dict)):
                return payload
            return [record for record in payload if record.get("split") == split]
        return payload
    raise ValueError(f"Unsupported dataset payload in {path}")


def _normalize_answers(record: dict[str, Any]) -> list[str]:
    if "golden_answers" in record and isinstance(record["golden_answers"], list):
        return [str(answer) for answer in record["golden_answers"]]
    if "answers" in record:
        answers = record["answers"]
        if isinstance(answers, list):
            return [str(answer) for answer in answers]
        return [str(answers)]
    if "answer" in record:
        answer = record["answer"]
        if isinstance(answer, list):
            return [str(item) for item in answer]
        return [str(answer)]
    raise ValueError(f"Could not infer answers from record: {record}")


def load_qa_examples(
    path: str | Path,
    *,
    split: str | None = None,
    max_samples: int | None = None,
    start_index: int = 0,
) -> list[DatasetExample]:
    file_path = Path(path)
    records = _load_raw_records(file_path, split)
    normalized: list[DatasetExample] = []
    for index, record in enumerate(records):
        qid = record.get("qid") or record.get("id") or f"{file_path.stem}-{index}"
        query = record.get("query") or record.get("question")
        if query is None:
            raise ValueError(f"Could not infer query from record: {record}")
        normalized.append(
            DatasetExample(
                qid=str(qid),
                query=str(query),
                golden_answers=_normalize_answers(record),
                context_paragraphs=[
                    ContextParagraph(
                        idx=int(paragraph.get("idx", paragraph_index)),
                        title=str(paragraph.get("title", "")).strip(),
                        contents=str(paragraph.get("paragraph_text", "")).strip(),
                    )
                    for paragraph_index, paragraph in enumerate(record.get("paragraphs", []))
                    if str(paragraph.get("title", "")).strip() and str(paragraph.get("paragraph_text", "")).strip()
                ],
            )
        )
    sliced = normalized[start_index:]
    if max_samples is not None:
        sliced = sliced[:max_samples]
    return sliced
