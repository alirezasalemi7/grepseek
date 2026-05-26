"""Sharded parallel-search engine for inference-time tool calls.

Inference-only. Wraps the shell-pipeline executor used by `inference.tools.run_tool`
so that "parallel-safe" pipelines (substring search, AND-narrow, head, wc -l,
sort | head) fan out across N pre-built shards instead of scanning the single
14GB corpus file. Pipelines the classifier can't safely rewrite fall back to
the existing single-file execution path — byte-identical output guaranteed
for both paths.

Toggle via `inference/run.py --engine_mode {inproc,daemon}`. The default
(`none`) disables the engine entirely; the inference pipeline runs exactly as
before (single-file path).

Public API:
    from inference.parallel_search import ShardedSearchEngine, Strategy

    engine = ShardedSearchEngine(shard_dir="/path/to/shards_16", n_shards=16,
                                  fallback_corpus_file="wiki_corpus.jsonl")
    stdout, returncode = engine.execute("rg -F 'foo' corpus.jsonl | head -n 3")

The RL trainer (rl/grepseek/trainer/verl_integration/search_corpus_tool.py) does
NOT use this engine — RL training stays on the single-file path.
"""

from .client import DaemonClient
from .engine import ShardedSearchEngine
from .pipeline import Strategy, classify, parse_pipeline

__all__ = ["ShardedSearchEngine", "DaemonClient", "Strategy",
           "parse_pipeline", "classify"]
