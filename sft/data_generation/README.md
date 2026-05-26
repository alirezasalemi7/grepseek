# GrepSeek — Cold-start SFT data generation

This module generates the **cold-start supervised fine-tuning (SFT) data** for
GrepSeek: multi-step trajectories in which an agent answers a multi-hop question
by issuing **shell / `ripgrep` commands** against a raw Wikipedia corpus and
reasoning over the results.

Each trajectory is built by a teacher LLM in three stages:

1. **Decompose** the question into ordered sub-questions.
2. **Backward construction** — for each sub-question (from last to first), search
   the corpus and verify, with an LLM judge, a command + passage that supports
   the known answer *without* using the answer as a search term.
3. **Forward assembly** — a planner drafts each step's reasoning and command from
   the question and the history *only* (it never sees the answer); a tutor then
   rewrites the reasoning so it leads to the verified command while staying
   grounded in what has actually been observed.

A final post-hoc **coherence judge** drops trajectories whose reasoning leaks
future facts or is otherwise inconsistent. The result is a clean
`think → command → observation → … → answer` trace per example.

> This is the public, simplified pipeline: a single trajectory per question, no
> self-correction injection and no multi-strategy sampling.

## 1. Install

```bash
pip install -r requirements.txt
```

You also need **ripgrep** (`rg`) on your `PATH` (the agent's search tool):

```bash
conda install -c conda-forge ripgrep      # or: apt-get install ripgrep / brew install ripgrep
```

## 2. Get the corpus

The agent searches the **Wikipedia-2018 corpus** used by Search-R1 (~21M
passages). Download and unpack it with the helper script:

```bash
python download_corpus.py --dest data/wiki_18
```

This fetches `wiki-18.jsonl.gz` (~5 GB) from
[`PeterJinGo/wiki-18-corpus`](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
and writes `data/wiki_18/wiki_corpus.jsonl` (~14 GB). Each line is one passage:

```json
{"id": "0", "contents": "\"Anarchism\"\nAnarchism is a political philosophy ..."}
```

> Make sure the destination **and** your HF cache (`HF_HOME`) are on a roomy
> filesystem — not your home directory.

Point the pipeline at that directory with `--corpus_dir data/wiki_18` (or the
`CORPUS_DIR` env var). The agent refers to the file as `corpus.jsonl`; the runner
rewrites that to `wiki_corpus.jsonl` at execution time.

> The QA questions (HotpotQA / NQ) are pulled automatically from
> `RUC-NLPIR/FlashRAG_datasets` via 🤗 `datasets` — no manual download needed.

## 3. Launch a teacher server

The teacher is **Qwen3.5-27B** (what the paper uses), served behind any
**OpenAI-compatible** endpoint. For example, with vLLM:

```bash
vllm serve Qwen/Qwen3.5-27B \
    --served-model-name teacher --host 0.0.0.0 --port 8000
```

Then tell the pipeline where it is (flags or env vars):

```bash
export LLM_HOST=127.0.0.1 LLM_PORT=8000 LLM_MODEL=teacher
```

(Any sufficiently capable instruct model works. To use the hosted OpenAI API
instead, point a small OpenAI-compatible proxy at it, or set
`OPENAI_API_KEY`/`base_url` in `utils/llm.py`.)

## 4. Generate

```bash
python create_data.py \
    --dataset hotpotqa --split train --n 100 \
    --corpus_dir /path/to/wiki_18 \
    --out output/traces.jsonl \
    --out_chatml output/sft.jsonl \
    --out_pretty output/pretty.txt \
    --parallel_examples 8
```

Or just run the example end-to-end (after exporting `LLM_MODEL` and `CORPUS_DIR`):

```bash
bash scripts/run_example.sh
```

### Outputs

| file | contents |
|---|---|
| `--out` (`traces.jsonl`) | one rich record per example: decomposition, backward steps, forward steps, final answer, F1/EM, judge verdict, and the rendered trajectory — for debugging. |
| `--out_chatml` (`sft.jsonl`) | the **SFT data**: `{"id", "question", "messages", ...}`, only for trajectories that are correct *and* pass the judge. |
| `--out_pretty` (`pretty.txt`) | human-readable dump (`question → think → command → result → answer`) for eyeballing. |

## 5. Build training files

Convert the SFT messages into train/val parquet:

```bash
python to_parquet.py --in 'output/sft.jsonl' --out_dir output/sft_parquet --include_tools
```

`--include_tools` attaches the `shell` function-call schema as a `tools` column.
The split is deterministic by `hash(id)`, so adding more data later keeps the
existing val membership stable.

## Useful flags

| flag | default | meaning |
|---|---|---|
| `--dataset` | `hotpotqa` | `hotpotqa` (multi-hop) or `nq` (mostly single-hop) |
| `--n` / `--start` | `20` / `0` | how many examples, and the slice offset |
| `--parallel_examples` | `9` | concurrent examples = max concurrent LLM calls |
| `--max_tool_calls` | `10` | hard cap on tool-call turns per trajectory |
| `--backward_max_iterations` | `6` | command-refinement tries per sub-question |
| `--no_quality_filter` | off | disable the post-hoc coherence judge |
| `--resume` / `--resume_from` | — | skip already-processed ids (resume / parallel ranges) |
| `--servers PATH` | — | optional multi-server pool (`servers.json`); see below |

### Multi-server pool (optional)

For large runs you can load-balance across several teacher servers. Create a
`servers.json`:

```json
{
  "reload_interval_s": 3600,
  "servers": [
    {"host": "node1", "port": 8000, "model": "teacher", "max_in_flight": 8},
    {"host": "node2", "port": 8000, "model": "teacher", "max_in_flight": 8}
  ]
}
```

and pass `--servers servers.json`. The file is hot-reloaded by mtime, per-server
in-flight caps are honored, failures route around dead servers, and calls for
the same example stick to one server to maximize prefix-cache hits.

## Environment variables

| var | used for |
|---|---|
| `LLM_HOST`, `LLM_PORT`, `LLM_MODEL` | default teacher endpoint |
| `CORPUS_DIR` | directory containing `wiki_corpus.jsonl` |
| `TOKENIZER_NAME` | tokenizer for capping tool output (falls back to a char estimate if unavailable) |
| `HF_DATASETS_CACHE` | optional 🤗 datasets cache location |

## Layout

```
sft/data_generation/
  create_data.py        # entry point
  to_parquet.py         # SFT jsonl -> train/val parquet
  scripts/run_example.sh
  utils/
    pipeline.py         # decomposition + backward + forward + judge
    prompts.py          # all teacher/planner/tutor/judge prompts
    llm.py              # OpenAI-compatible client
    server_pool.py      # optional multi-server pool
    tools.py            # sandboxed shell/ripgrep executor
    load_hotpotqa.py    # HotpotQA loader (FlashRAG_datasets)
    load_nq.py          # NQ-Open loader (FlashRAG_datasets)
```
