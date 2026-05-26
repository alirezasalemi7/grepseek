# GrepSeek — Inference (generation + evaluation)

Run a trained GrepSeek model as a search agent: it answers questions by issuing
`rg`/`grep` shell-pipeline tool calls against the Wikipedia corpus, against a
served (OpenAI-compatible) vLLM endpoint. Two modes:

- **Generation** — run the agent on your own questions and write the trajectories
  + answers. No benchmark, gold answers optional, no scoring required.
- **Benchmark eval** — run + score on the Search-R1 QA suite (NQ, TriviaQA, PopQA,
  HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle), reporting EM and F1.

It also ships the **fast execution engine** ([`parallel_search/`](parallel_search))
— sharded parallel grep with an optional search daemon — that speeds up the tool
calls without changing their output (byte-identical to the single-file path).

Everything is plain Python (`python -m inference.run`); nothing SLURM-specific.

## Prerequisites

1. **Environment** — the training/inference env from [`../TRAINING_ENV.md`](../TRAINING_ENV.md)
   (needs `vllm`, `transformers`, `openai`, `datasets`).
2. **A served model** — bring up your checkpoint with [`../rl/serve_rl.sh`](../rl/serve_rl.sh)
   (or any OpenAI-compatible vLLM server with Qwen3 tool calling). Note its
   `--base_url` (e.g. `http://HOST:PORT/v1`) and served model name.
3. **Corpus** — `wiki_corpus.jsonl`; point `--corpus_dir` (or `$GREPSEEK_CORPUS_ROOT`)
   at the directory that holds it. Download with
   [`../cold_start_sft/download_corpus.py`](../cold_start_sft/download_corpus.py).

## Generation (eval optional)

Input is a JSON/JSONL where each row has a `question` (or `query`); `id` and
`golden_answers` are optional:

```jsonl
{"id": "q1", "question": "Who wrote the novel that inspired the film Blade Runner?"}
{"question": "What year did the Apollo 11 mission land on the moon?"}
```

```bash
GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus \
  bash run_inference.sh \
    --base_url http://HOST:PORT/v1 --model grepseek \
    --input my_questions.jsonl --out_dir output/gen
```

Writes `output/gen/<input-stem>/predictions.jsonl` (one trajectory per line:
question, full turn-by-turn messages, tool calls, final `prediction`, timing) and
`summary.json` (counts + timing). If some rows carry `golden_answers` they are
scored too; pass `--no_eval` to never score.

## Benchmark eval

```bash
GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus \
  bash run_inference.sh \
    --base_url http://HOST:PORT/v1 --model grepseek \
    --datasets nq,hotpotqa,2wikimultihopqa,bamboogle --limit 200 \
    --parallel 16 --out_dir output/eval
```

Per dataset it downloads the FlashRAG split, runs the agent in parallel, scores
each trajectory with normalized **EM** and token-overlap **F1**, and writes
`<dataset>/predictions.jsonl` + `summary.json`. At the end it prints a table and
writes `overall.json` with both **macro** (datasets weighted equally) and
**micro** (examples pooled across datasets) EM/F1. `--datasets all` runs the full
suite. EM and F1 are reported separately (no combined metric).

## Fast execution engine (optional speedup)

Each tool call greps a 14 GB corpus; over thousands of calls this dominates
wall-clock. The engine fans "parallel-safe" pipelines (substring search,
AND-narrow, `head`, `wc -l`, `sort | head`) across N pre-built shards; anything
it can't safely rewrite falls back to the single-file path — **byte-identical
output guaranteed** (see [`parallel_search/tests/`](parallel_search/tests)).

```bash
# 1. Build shards once (idempotent; re-run is a no-op if the manifest matches):
python -m inference.parallel_search.sharder \
    --src /path/to/wiki_18_corpus/wiki_corpus.jsonl --dst /path/to/shards_16 --n 16

# 2a. In-process engine — pass --engine_mode inproc + --shard_dir:
bash run_inference.sh ... --engine_mode inproc --shard_dir /path/to/shards_16

# 2b. Or a shared daemon (one process serves many eval workers over a socket):
python -m inference.parallel_search.daemon --shard_dir /path/to/shards_16 \
    --socket /tmp/grepseek_search.sock &
bash run_inference.sh ... --engine_mode daemon --socket /tmp/grepseek_search.sock
```

Default is `--engine_mode none` (single-file path); the engine is purely an
optimization. `--engine_log <path.jsonl>` records per-call telemetry.

## Package layout

```
inference/
├── run.py                  # the harness: generation + (optional) eval  (python -m inference.run)
├── run_inference.sh        # thin launcher (sets PYTHONPATH + caches)
├── agent.py                # multi-turn search agent (one trajectory per question)
├── tools.py                # shell/grep tool executor + validator
├── load_dataset.py         # FlashRAG benchmark loaders + generic questions-file loader
├── scoring.py              # normalize / EM / token-F1 / aggregate
└── parallel_search/        # the fast execution engine
    ├── sharder.py          # build N shards from the corpus
    ├── daemon.py, client.py # shared search daemon + client
    ├── engine.py, pipeline.py, runners.py
    └── tests/              # byte-equivalence + daemon unit tests (run: python -m pytest parallel_search/tests)
```
