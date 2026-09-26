"""PACE: Pruned Adaptive Command Execution.

A one-time, offline auxiliary structure (vocabulary suffix array +
Elias-Fano posting lists + separator-rank bitvector) over a flat corpus,
used to narrow literal `rg`/`grep` pipelines to candidate lines before the
real command runs. PACE never produces output itself: it only ever
narrows the input of the real pipeline, and the caller always falls back
to running the original, unmodified command whenever narrowing cannot be
proven safe. See `core` for the full correctness contract and cost model.
"""
from .core import (
    PACE_DIR_ENV_VAR,
    PaceBudget,
    PaceDecision,
    PaceMetadata,
    PacePlan,
    PaceStructure,
    StalePaceError,
    build_pace,
    classify_for_pace,
    decide_for_pace,
    load_pace_from_env,
    split_pipeline_stages,
    verify_natively,
    verify_natively_streaming,
)

__all__ = [
    "PACE_DIR_ENV_VAR",
    "PaceBudget",
    "PaceDecision",
    "PaceMetadata",
    "PacePlan",
    "PaceStructure",
    "StalePaceError",
    "build_pace",
    "classify_for_pace",
    "decide_for_pace",
    "load_pace_from_env",
    "split_pipeline_stages",
    "verify_natively",
    "verify_natively_streaming",
]
