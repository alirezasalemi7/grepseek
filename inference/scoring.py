"""QA scoring metrics — EM and F1 reported separately.

Mirrors the SQuAD-style normalization used in
`cold_start_data_gen/utils/pipeline.py` so that evaluation numbers are directly
comparable to training-time success rates and reward signals.

Public API:
    normalize_answer(s)               -> str
    exact_match(pred, gold_answers)   -> bool
    f1_score(pred, gold_answers)      -> float
    score(pred, gold_answers)         -> {"em": bool, "f1": float}
    aggregate(scores)                 -> {"em": mean, "f1": mean, "n": int}

Both metrics use max-over-gold semantics — we score the prediction against the
best gold answer, not the average. This matches the FlashRAG / HotpotQA /
SQuAD convention.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, Union

_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _coerce_golds(gold_answers: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(gold_answers, str):
        return [gold_answers]
    return [g for g in gold_answers if g]


def exact_match(prediction: str, gold_answers: Union[str, Iterable[str]]) -> bool:
    if not prediction:
        return False
    pred_n = normalize_answer(prediction)
    return any(pred_n == normalize_answer(g) for g in _coerce_golds(gold_answers))


def _f1_against(pred_norm: str, gold_norm: str) -> float:
    """SQuAD-style token-overlap F1 between two normalized strings."""
    if not pred_norm or not gold_norm:
        return 1.0 if pred_norm == gold_norm else 0.0
    pred_toks = pred_norm.split()
    gold_toks = gold_norm.split()
    if not pred_toks or not gold_toks:
        return 0.0
    pred_count: dict[str, int] = {}
    for t in pred_toks:
        pred_count[t] = pred_count.get(t, 0) + 1
    num_same = 0
    for t in gold_toks:
        if pred_count.get(t, 0) > 0:
            pred_count[t] -= 1
            num_same += 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, gold_answers: Union[str, Iterable[str]]) -> float:
    """Max F1 across gold answers, after SQuAD-style normalization."""
    if not prediction:
        return 0.0
    pred_norm = normalize_answer(prediction)
    best = 0.0
    for g in _coerce_golds(gold_answers):
        v = _f1_against(pred_norm, normalize_answer(g))
        if v > best:
            best = v
    return best


def score(prediction: str, gold_answers: Union[str, Iterable[str]]) -> dict:
    """Return {"em": bool, "f1": float} for a single prediction."""
    return {
        "em": exact_match(prediction, gold_answers),
        "f1": f1_score(prediction, gold_answers),
    }


def aggregate(scores: list[dict]) -> dict:
    """Mean EM and mean F1 over a list of per-example scores.

    Returns 0.0 for an empty list so callers don't have to guard against it.
    """
    n = len(scores)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "n": 0}
    em_sum = sum(1 for s in scores if s.get("em"))
    f1_sum = sum(float(s.get("f1") or 0.0) for s in scores)
    return {"em": em_sum / n, "f1": f1_sum / n, "n": n}
