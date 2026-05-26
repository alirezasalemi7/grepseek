# GrepSeek — Supervised fine-tuning (cold start)

Fine-tune the base model on the cold-start trajectories produced by
[`../cold_start_sft`](../cold_start_sft). This is the SFT stage that initializes
the policy before RL. Training uses verl's FSDP SFT trainer
(`verl.trainer.sft_trainer`).

## Prerequisites

1. **verl** — vendored at the repo root in [`../verl`](../verl) (the exact
   version used in the paper; nothing to initialize). `run_sft.sh` puts it on
   `PYTHONPATH` automatically.

2. The training environment (CUDA 12.8, PyTorch 2.10, flash-attn, vLLM, …).
   Set it up from the exact snapshots in [`../TRAINING_ENV.md`](../TRAINING_ENV.md)
   (conda `environment-train.yml` or the pip freeze). Multi-GPU (e.g. 4×A100-80GB)
   is recommended for the 9B model at 16K sequence length.

3. **Training data** — run the data-gen + parquet steps first:

   ```bash
   cd ../cold_start_sft
   python create_data.py --dataset hotpotqa --n 10000 --out_chatml output/sft.jsonl ...
   python to_parquet.py --in 'output/sft.jsonl' --out_dir output/sft_parquet --include_tools
   ```

## Run

Runs on **any machine with ≥2 GPUs** (no cluster/scheduler required) —
`run_sft.sh` is a plain `torchrun` launcher. Activate your training env (see
[`../TRAINING_ENV.md`](../TRAINING_ENV.md)), then:

```bash
export TRAIN_PARQUET=../cold_start_sft/output/sft_parquet/train.parquet
export MODEL_PATH=Qwen/Qwen3.5-9B     # base model to fine-tune
export NPROC=4                         # number of GPUs (4×A100-80GB reproduces the paper)
bash run_sft.sh
```

Checkpoints are written to `SAVE_DIR` (default `../checkpoints/sft/<timestamp>/`).
All knobs are environment variables — see the top of [`run_sft.sh`](run_sft.sh).

> **GPU count / sequence parallelism.** The 9B model at 16K sequence length needs
> sequence parallelism, or each rank momentarily materializes an unsharded ~37 GB
> fp32 layer unit during the forward all-gather and OOMs even on 80 GB cards.
> `run_sft.sh` therefore defaults `ULYSSES_SP_SIZE` to `NPROC`. The paper used
> `NPROC=4`, `ULYSSES_SP_SIZE=4`; this is the validated config. `ULYSSES_SP_SIZE`
> must divide `NPROC` (and the model's attention-head count) — for an unusual GPU
> count, set it explicitly to a divisor. A single GPU cannot hold the 9B (FSDP has
> nothing to shard across), so use **≥2 GPUs, ≥4 recommended**.

## Hyperparameters (paper defaults)

| setting | value | env var |
|---|---|---|
| base model | Qwen3.5-9B | `MODEL_PATH` |
| epochs | 1 | `TOTAL_EPOCHS` |
| learning rate | 5e-6 | `LR` |
| weight decay | 0.01 | `WEIGHT_DECAY` |
| warmup ratio | 0.05 | `WARMUP_RATIO` |
| max sequence length | 16384 | `MAX_LENGTH` |
| global batch size | 32 | `TRAIN_BATCH_SIZE` |
| precision / strategy | bf16, FSDP (param+optimizer offload) | — |
| GPUs | 4×A100-80GB | `NPROC` |
| seq. parallelism | NPROC (paper: 4) | `ULYSSES_SP_SIZE` |

> **1 epoch on purpose.** A 2-epoch run caused entropy collapse that hurt the
> downstream RL stage (≈55% of GRPO groups gave flat advantage). One epoch is
> ample for the JSON+tag format while preserving rollout diversity for RL.

## Smoke test (validate wiring without a full run)

Run a 5-step, no-checkpoint pass with the production config to confirm the data,
chat-template, and FSDP wiring before committing to a full run. `TOTAL_TRAINING_STEPS`
caps the steps and `SAVE_FREQ=-1` skips checkpoints:

```bash
TRAIN_PARQUET=.../train.parquet MODEL_PATH=Qwen/Qwen3.5-9B \
NPROC=4 ULYSSES_SP_SIZE=4 \
TOTAL_TRAINING_STEPS=5 SAVE_FREQ=-1 SAVE_DIR=/tmp/sft_smoke \
bash run_sft.sh
```

Expect a falling loss over the 5 steps (we observed `0.81 → 0.65 → 0.55`) and no
`input_ids` mismatch assertion (`data.ignore_input_ids_mismatch=true` tolerates
Qwen3 thinking-tag re-rendering). Use the same `NPROC`/`ULYSSES_SP_SIZE` as your
real run — a single GPU cannot hold the 9B, so this smoke also needs ≥2 GPUs.

## Output → next stage

verl writes sharded **FSDP checkpoints** under `SAVE_DIR`. To use the model for
inference or as the RL initialization, merge it to HuggingFace format with verl's
model merger:

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir SAVE_DIR/global_step_<N> --target_dir SAVE_DIR/hf
```

The merged HF checkpoint is the starting policy for the RL stage (`../rl`).
