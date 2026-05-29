# GrepSeek: Training Search Agents for Direct Corpus Interaction

[![Paper](https://img.shields.io/badge/arXiv-2605.29307-b31b1b.svg)](https://arxiv.org/abs/2605.29307)
[![Models](https://img.shields.io/badge/🤗%20Models-grepseek-yellow.svg)](https://huggingface.co/collections/alireza7/grepseek)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ColdStart--SFT--10k-yellow.svg)](https://huggingface.co/datasets/alireza7/GrepSeek-ColdStart-SFT-10k)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**GrepSeek** trains a compact open-weight LLM (Qwen3.5-9B) to answer
knowledge-intensive questions by **searching a raw text corpus with Unix shell
commands** (`rg`, `grep`, `head`, …) — *Direct Corpus Interaction (DCI)* — instead
of retrieving from a pre-computed dense or sparse index. Coupling retrieval and
reasoning in one policy gives exact lexical control (great for entity
disambiguation and multi-hop reasoning), needs **no embedding index** (just the
~14 GB raw corpus), and — with our **semantics-preserving sharded-parallel
execution engine** — runs corpus search up to **7.6× faster** while staying
**byte-exact** with plain `grep`.

<p align="center"><img src="assets/method_figure.png" width="780" alt="Retrieval-augmented agentic search vs. Direct Corpus Interaction"></p>

## Highlights

- **Two-stage training.** A cold-start SFT dataset built by an *Answer-Aware
  Tutor + Answer-Blind Planner* pipeline (causally-consistent, leakage-free search
  trajectories), then **GRPO** with a token-F1 × format-gate reward.
- **Best overall accuracy.** Highest micro-average **F1 0.5691 / EM 0.4948** over
  7 QA benchmarks, beating strong RL-trained dense/sparse retrieval agents.
- **Index-free & cheap to stand up.** 14 GB RAM (the raw corpus) vs. 70 GB (E5) /
  221 GB (Qwen3-4B-Emb); ~1 min setup vs. 3.2 / 62.4 A100-hours of offline indexing.
- **Full, runnable release.** Cold-start data generation, SFT, RL (GRPO), and the
  inference harness + fast execution engine — all reproducible end-to-end.

## Results (token-level F1)

Trained only on NQ + HotpotQA (marked \*); the other five datasets are
out-of-distribution. GrepSeek wins 4/7 and the best micro-average.

| Method | NQ\* | TriviaQA | PopQA | HotpotQA\* | 2Wiki | MuSiQue | Bamboogle | **micro-avg** |
|---|---|---|---|---|---|---|---|---|
| Search-R1 (Qwen3-4B-Emb, best baseline) | 0.5067 | **0.7693** | **0.5101** | 0.5591 | 0.4299 | 0.2878 | **0.6989** | 0.5441 |
| **GrepSeek** | **0.5223** | 0.7673 | 0.4861 | **0.6231** | **0.5178** | **0.3006** | 0.6212 | **0.5691** |

(EM table and full baselines in the paper.)

## Released artifacts

| | HuggingFace |
|---|---|
| RL model (final GrepSeek) | [`alireza7/GrepSeek-Qwen3.5-9B-GRPO`](https://huggingface.co/alireza7/GrepSeek-Qwen3.5-9B-GRPO) |
| Cold-start SFT model | [`alireza7/GrepSeek-Qwen3.5-9B-SFT`](https://huggingface.co/alireza7/GrepSeek-Qwen3.5-9B-SFT) |
| Cold-start SFT dataset (10k) | [`alireza7/GrepSeek-ColdStart-SFT-10k`](https://huggingface.co/datasets/alireza7/GrepSeek-ColdStart-SFT-10k) |
| Corpus (2018 Wikipedia, 21M passages) | [`PeterJinGo/wiki-18-corpus`](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) |

## Repository layout

```
grepseek/
├── containers/       # Docker/GHCR and Apptainer usage instructions (no large artifacts tracked)
├── sft/              # cold-start SFT data generation + supervised fine-tuning
├── rl/               # GRPO training + serving + checkpoint merge  (the `grepseek` package)
├── inference/        # the agent harness (generation + eval) + the fast parallel-search engine
├── notebooks/        # interactive Colab/local demo notebooks (try the released model without writing code)
├── verl/             # vendored training engine (Apache-2.0; see verl/VENDORED.md)
├── TRAINING_ENV.md   # exact, verified environment recipe (CUDA 12.8 / torch 2.10 / vLLM 0.17 / …)
└── examples/         # sample questions for the inference quickstart
```
Each stage has its own README: [SFT data generation](sft/data_generation),
[SFT training](sft), [rl](rl), [inference](inference).

## Installation

```bash
git clone https://github.com/alirezasalemi7/grepseek && cd grepseek
```
All commands below assume this repository root as the current working directory.
Set up the training/inference environment from the **portable, verified recipe**
in [`TRAINING_ENV.md`](TRAINING_ENV.md) (conda CUDA 12.8 toolkit → torch 2.10
cu128 → flash-attn → vLLM 0.17 → verl via `PYTHONPATH`). The lightweight
data-generation stage has its own smaller env — see
[sft/data_generation/README.md](sft/data_generation/README.md).

Alternatively, use the prebuilt Docker/GHCR or Apptainer environments described
in [`containers/README.md`](containers/README.md). The container images include
the runtime environments only; mount this repository, data, cache, and
checkpoints at runtime.

Download the corpus once (≈14 GB; verified byte-identical to the paper's):
```bash
python sft/data_generation/download_corpus.py --dest data/wiki_18_corpus
```

## Quickstart — run the released model (no training)

Serve the final checkpoint and ask it questions. Models load straight from the
Hub (add `HF_TOKEN=...` if the repo is private).

```bash
# 1. serve GrepSeek (1–2 A100s; vLLM, OpenAI-compatible, Qwen3 tool calling)
MODEL_PATH=alireza7/GrepSeek-Qwen3.5-9B-GRPO TP_SIZE=1 bash rl/serve_rl.sh  # -> http://localhost:10730/v1

# 2a. generation on your own questions
GREPSEEK_CORPUS_ROOT=data/wiki_18_corpus \
  bash inference/run_inference.sh --base_url http://localhost:10730/v1 \
    --model grepseek --temperature 0.6 --input examples/questions.jsonl --out_dir output/gen

# 2b. reproduce the benchmark eval (token-F1 / EM on the 7-dataset suite)
GREPSEEK_CORPUS_ROOT=data/wiki_18_corpus \
  bash inference/run_inference.sh --base_url http://localhost:10730/v1 \
    --model grepseek --temperature 0.6 --datasets all --out_dir output/eval
```
Add the **fast execution engine** (sharded parallel grep / daemon) for a large
speedup — see [inference/README.md](inference/README.md).

## Notebooks (interactive demo)

[`notebooks/GrepSeek_demo.ipynb`](notebooks/GrepSeek_demo.ipynb) is a
self-contained Jupyter demo that runs **on Google Colab *or* on a local CUDA
box** — Step 0 auto-detects which. On Colab it installs `ripgrep` + vLLM + the
Qwen3.5 transformers build and clones the repo; locally it reuses your
existing env / repo / corpus / checkpoint. The notebook spins up vLLM in the
runtime and prints the agent's full `think → tool_call → tool_response →
answer` trajectory. A `SERVE_MODE = "4bit"` toggle (on-the-fly bitsandbytes)
lets a free **T4 (16 GB)** host the 9B model. Nothing is hard-coded —
`GREPSEEK_REPO`, `GREPSEEK_MODEL`, `GREPSEEK_CORPUS_ROOT`, `GREPSEEK_HOST`,
`GREPSEEK_PORT` all override the defaults.

- **On Colab:** click the badge, set *Runtime → Change runtime type → GPU*
  (A100 / L4 / T4), then **Run all**. Step 1 handles installs; Step 2
  downloads the ~14 GB corpus.
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alirezasalemi7/grepseek/blob/main/notebooks/GrepSeek_demo.ipynb)
- **Locally** (CUDA host with the env from [`TRAINING_ENV.md`](TRAINING_ENV.md)
  already activated, repo cloned, corpus downloaded):
  ```bash
  GREPSEEK_CORPUS_ROOT=data/wiki_18_corpus jupyter notebook notebooks/GrepSeek_demo.ipynb
  ```
  Step 1 detects you're off-Colab and skips apt/pip/clone; Step 2 finds the
  corpus and skips the download.

## Reproduce the full pipeline from scratch

```bash
# 1. Cold-start SFT data — Answer-Aware Tutor + Answer-Blind Planner (serve a Qwen3.5-27B teacher first)
#    -> see sft/data_generation/README.md ; or skip and use the released dataset.
python sft/data_generation/create_data.py ...                                     # details in its README
python sft/data_generation/to_parquet.py --in 'sft/data_generation/output/sft.jsonl' \
  --out_dir sft/data_generation/output/sft_parquet --include_tools

# 2. Supervised fine-tuning (4×A100-80GB)
#    -> sft/README.md
TRAIN_PARQUET=.../train.parquet MODEL_PATH=Qwen/Qwen3.5-9B NPROC=4 bash sft/run_sft.sh
PYTHONPATH=$PWD/verl:${PYTHONPATH:-} python -m verl.model_merger merge \
  --backend fsdp --local_dir <sft_ckpt>/global_step_N \
  --target_dir <sft_ckpt>/global_step_N/huggingface

# 3. RL with GRPO (4×A100-80GB), initialized from the SFT checkpoint
#    -> rl/README.md
python rl/prepare_rl_data.py --out_dir data/rl/nq_hotpot          # NQ + HotpotQA
GREPSEEK_MODEL_PATH=<sft_ckpt>/global_step_N/huggingface \
GREPSEEK_TRAIN_FILES=data/rl/nq_hotpot/train.jsonl \
GREPSEEK_VAL_FILES=data/rl/nq_hotpot/dev.jsonl GREPSEEK_CORPUS_ROOT=data/wiki_18_corpus \
NPROC=4 bash rl/run_rl.sh
CKPT_DIR=<rl_ckpt>/global_step_200 bash rl/convert_to_hf.sh       # merge for serving

# 4. Evaluate — see Quickstart step 2b.
```
Each stage's README has the exact knobs and hyperparameters.

## Citation

```bibtex
@misc{salemi2026grepseektrainingsearchagents,
      title={GrepSeek: Training Search Agents for Direct Corpus Interaction},
      author={Alireza Salemi and Chang Zeng and Atharva Nijasure and Jui-Hui Chung and Razieh Rahimi and Fernando Diaz and Hamed Zamani},
      year={2026},
      eprint={2605.29307},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.29307},
}
```

## License & acknowledgements

Code is released under the [Apache-2.0](LICENSE) license. The model weights derive
from `Qwen/Qwen3.5-9B` and are subject to its license. Training uses a vendored
copy of [verl](https://github.com/volcengine/verl) (Apache-2.0; see
[verl/VENDORED.md](verl/VENDORED.md)). Benchmarks are from the
[FlashRAG](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets) suite and
the corpus from [`PeterJinGo/wiki-18-corpus`](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus).

This work was supported in part by the Center for Intelligent Information Retrieval, in part by the Office of Naval Research contract \#N000142412612, in part by the National Science Foundation grant \#2402873 and \#2402874, and with support from Google.org. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsors.
