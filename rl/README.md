# GrepSeek — RL (GRPO)

Reinforcement-learning stage: GRPO over the cold-start SFT policy, where the
model searches a 21M-passage Wikipedia corpus by issuing `rg`/`grep` shell
pipelines and is rewarded for answer correctness (F1). Training uses
[verl](../verl) as the engine; everything GrepSeek-specific lives in the
`grepseek` package here and is wired into verl purely through config:

| component | file |
|---|---|
| agent loop (multi-turn search) | [`grepseek/trainer/verl_integration/agent_loop.py`](grepseek/trainer/verl_integration/agent_loop.py) |
| search / grep tool (over the corpus) | [`grepseek/trainer/verl_integration/search_corpus_tool.py`](grepseek/trainer/verl_integration/search_corpus_tool.py) |
| reward (format-gated EM/F1 − length penalty) | [`grepseek/trainer/verl_integration/reward_function.py`](grepseek/trainer/verl_integration/reward_function.py) |
| dataset adapter (QA JSONL) | [`grepseek/trainer/verl_integration/dataset.py`](grepseek/trainer/verl_integration/dataset.py) |
| GRPO config (the paper's "ours" run) | [`grepseek/trainer/config/grpo_trainer.yaml`](grepseek/trainer/config/grpo_trainer.yaml) |

Both `grepseek` (this dir) and `verl` are used **via `PYTHONPATH`** — nothing to
pip-install. `run_rl.sh` sets the path up for you.

## Prerequisites

1. **Environment** — the training env from [`../TRAINING_ENV.md`](../TRAINING_ENV.md)
   (CUDA 12.8, PyTorch 2.10, vLLM 0.17, flash-attn, …). Same env as SFT.
2. **Starting policy** — a cold-start SFT checkpoint in **HuggingFace format**,
   set via `GREPSEEK_MODEL_PATH`. This can be **either**:
   - a **local HF-format directory** — train with [`../sft`](../sft), then merge the
     FSDP shards to HF
     (`python -m verl.model_merger merge --backend fsdp --local_dir <sft_ckpt>/global_step_NNN --target_dir <...>/huggingface`), or
   - a **HuggingFace Hub model ID** (e.g. `your-org/grepseek-sft-9b`) — verl and vLLM
     both load `model.path` straight from the Hub. For a **private** repo, also export
     `HF_TOKEN=...` so the download is authorized.

   If `GREPSEEK_MODEL_PATH` is unset, RL starts from the **base** model (ablation,
   not the paper result).
3. **Corpus** — `wiki_corpus.jsonl` (21M passages). Download with
   [`../sft/data_generation/download_corpus.py`](../sft/data_generation/download_corpus.py);
   point `GREPSEEK_CORPUS_ROOT` at the directory that contains it.
4. **RL data** — build it with one command from NQ + HotpotQA (the paper's data):
   ```bash
   python prepare_rl_data.py --out_dir data/rl/nq_hotpot     # writes train.jsonl + dev.jsonl
   ```
   Any QA JSONL works, though — one record per line with `id`/`qid`,
   `question`/`query`, and `golden_answers` (FlashRAG-style), e.g.
   `{"id": "nq_123", "question": "who wrote ...", "golden_answers": ["..."]}`. See
   [`prepare_rl_data.py`](prepare_rl_data.py) for subsampling options.

## Run

GrepSeek RL runs on **any machine with ≥2 GPUs** (no cluster/scheduler required).
Activate your training env (see [`../TRAINING_ENV.md`](../TRAINING_ENV.md)), set
five environment variables, and go:

```bash
export GREPSEEK_MODEL_PATH=/path/to/sft_ckpt/global_step_NNN/huggingface   # starting policy (local dir OR an HF Hub ID, e.g. your-org/grepseek-sft-9b)
export GREPSEEK_TRAIN_FILES=data/rl/nq_hotpot/train.jsonl
export GREPSEEK_VAL_FILES=data/rl/nq_hotpot/dev.jsonl
export GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus      # dir holding wiki_corpus.jsonl
export NPROC=4                                           # GPUs (paper: 4×A100-80GB)
bash run_rl.sh
```

Extra args pass straight through to verl, e.g.
`bash run_rl.sh actor_rollout_ref.actor.optim.lr=2e-6 trainer.total_training_steps=100`.
Checkpoints land in `GREPSEEK_OUTPUT_DIR` (default `../checkpoints/rl/<timestamp>/`).

To enable Weights & Biases, set `WANDB_API_KEY` and `WANDB_ENTITY` (otherwise it
logs to console only).

> **Corpus speed.** Each rollout runs `rg` over the 14 GB corpus. On a shared/
> network filesystem this is slow — copy `wiki_corpus.jsonl` to local disk first
> and point `GREPSEEK_CORPUS_ROOT` there.

## Hyperparameters (paper "ours")

These are the **defaults baked into `grpo_trainer.yaml`** — running `run_rl.sh`
reproduces the paper config; you only supply paths + GPU count.

| setting | value | where |
|---|---|---|
| algorithm | GRPO (no critic, no KL) | `algorithm.adv_estimator=grpo` |
| rollouts / question | 5 | `actor_rollout_ref.rollout.n` |
| reward metric | **F1** (EM also logged) | `custom_reward_function.reward_kwargs.reward_metric` |
| length penalty | **off** | `...reward_kwargs.enable_length_decay=false` |
| train batch (questions) | 256 | `data.train_batch_size` |
| PPO mini-batch | 32 | `actor.ppo_mini_batch_size` |
| PPO epochs | 1 | `algorithm.ppo_epochs` |
| learning rate | 5e-6, constant | `actor.optim.lr` |
| total steps | 200 (save every 40) | `trainer.total_training_steps` |
| context | 1024 prompt + 15360 response = 16K | `data.max_prompt_length` / `max_response_length` |
| max tool calls | 5 (+1 final answer) | `rollout.multi_turn.max_user_turns` |
| GPUs / seq-parallel | 4 / ulysses_sp=2 | `NPROC` / `ULYSSES_SP` |
| precision / strategy | bf16, FSDP (param+grad+optim offload) | — |

The reward is **format-gated**: a rollout scores 0 unless its transcript follows
the SFT-learned `<think>/<tool_call>/<tool_response>/<answer>` structure; valid
rollouts then score F1 against the gold answers. To turn the length penalty back
on, set `custom_reward_function.reward_kwargs.enable_length_decay=true` (tune `a`,
`decay_type`, `penalty_mode` — see the comments in `grpo_trainer.yaml`).

## Output → merge → serve

verl writes sharded FSDP checkpoints under `GREPSEEK_OUTPUT_DIR/global_step_NNN/actor/`.

```bash
# 1. Merge the actor FSDP shards into a single HF-format dir
CKPT_DIR=../checkpoints/rl/<run>/global_step_200 bash convert_to_hf.sh
#    -> writes <...>/global_step_200/actor/huggingface

# 2. Serve it with vLLM (OpenAI-compatible API, Qwen3 tool calling)
MODEL_PATH=../checkpoints/rl/<run>/global_step_200/actor/huggingface bash serve_rl.sh
#    -> http://<host>:10730/v1
```

## Package layout

```
rl/
├── grepseek/                       # importable package (PYTHONPATH=rl)
│   ├── prompting.py                # system-prompt assembly + final-answer prompt
│   ├── eval/scoring.py             # normalize / exact_match / token-F1
│   ├── data/qa.py                  # generic QA-JSONL loader (load_qa_examples)
│   └── trainer/
│       ├── __init__.py             # registers `grepseek_agent` on import
│       ├── verl_integration/       # agent loop, tool, reward, dataset, invalid-call handling
│       └── config/                 # grpo_trainer.yaml, agent_loops.yaml, tools/search_tools.yaml
├── prompts/                        # system_prompt.txt (+ two intentionally-empty template slots)
├── prepare_rl_data.py              # build train/dev JSONL from NQ + HotpotQA
├── run_rl.sh                       # GRPO launcher
├── convert_to_hf.sh                # FSDP → HF checkpoint merge
└── serve_rl.sh                     # vLLM serving of a trained checkpoint
```
