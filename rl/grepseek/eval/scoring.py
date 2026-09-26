from __future__ import annotations

import string
from collections import Counter


_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    tokens = [token for token in without_punctuation.split() if token not in _ARTICLES]
    return " ".join(tokens)


def exact_match_score(prediction: str, gold_answers: list[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(answer) for answer in gold_answers)


def token_f1_score(prediction: str, gold_answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1_score(prediction: str, gold_answers: list[str]) -> float:
    return max((token_f1_score(prediction, gold) for gold in gold_answers), default=0.0)
