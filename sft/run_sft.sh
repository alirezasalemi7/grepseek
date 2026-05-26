#!/usr/bin/env bash
# Supervised fine-tuning of the base model on the cold-start trajectories, via
# verl's FSDP SFT trainer (`verl.trainer.sft_trainer`). Reads the train.parquet
# produced by data_generation/to_parquet.py.
#
# Quick start:
#   export TRAIN_PARQUET=/path/to/sft_parquet/train.parquet
#   export MODEL_PATH=Qwen/Qwen3.5-9B          # base model to fine-tune
#   export NPROC=4                              # number of GPUs
#   bash run_sft.sh
#
# All knobs below are overridable via environment variables. Runs on any machine
# with >=2 GPUs once your training env is active (see ../TRAINING_ENV.md).
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

# --- verl on PYTHONPATH (the `verl/` git submodule at the repo root) ---
VERL_DIR="${VERL_DIR:-${REPO_ROOT}/verl}"
if [[ ! -d "${VERL_DIR}/verl/trainer" ]]; then
  echo "error: verl not found at ${VERL_DIR}. Initialize the submodule:" >&2
  echo "       git -C ${REPO_ROOT} submodule update --init --recursive verl" >&2
  exit 1
fi
export PYTHONPATH="${VERL_DIR}:${PYTHONPATH:-}"

# --- caches (keep off the home directory) ---
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
# Triton autotuner races on shared/parallel filesystems across torchrun ranks;
# pin its cache to local scratch (override TRITON_CACHE_DIR for node-local disk).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$$}"
mkdir -p "${TRITON_CACHE_DIR}"
# Let the CUDA allocator grow segments (helps with dynamic batch sizes).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=true

# --- config (override via env) ---
TRAIN_PARQUET="${TRAIN_PARQUET:?set TRAIN_PARQUET to .../sft_parquet/train.parquet from sft/data_generation/to_parquet.py}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NPROC="${NPROC:-1}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/sft/$(date +%Y%m%d-%H%M%S)}"

MAX_LENGTH="${MAX_LENGTH:-16384}"           # raise if your data has longer sequences
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"  # global batch (grad-accumulated)
MICRO_BATCH_PER_GPU="${MICRO_BATCH_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"
# Sequence parallelism. The 9B at 16K OOMs on the forward parameter all-gather
# WITHOUT it (each rank briefly materializes an unsharded ~37GB fp32 layer unit),
# so the paper used ulysses_sp = NPROC (4 on 4xA100). Must divide NPROC and the
# model's attention-head count. Defaults to NPROC to reproduce the paper; set to 1
# only if you have enough per-GPU memory to skip sequence parallelism.
ULYSSES_SP_SIZE="${ULYSSES_SP_SIZE:-${NPROC}}"
LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"           # 1 epoch keeps rollout diversity for the later RL stage
SAVE_FREQ="${SAVE_FREQ:-50}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"  # set (e.g. 5) to cap steps for a smoke test; empty = full epochs

# --- optional Weights & Biases (console-only unless BOTH a key and an entity are set) ---
# verl errors out if wandb is enabled without an entity, so require WANDB_ENTITY too.
if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_ENTITY:-}" ]]; then
  LOGGER='[console,wandb]'
  export WANDB_PROJECT="${WANDB_PROJECT:-grepseek-sft}"
else
  [[ -n "${WANDB_API_KEY:-}" ]] && echo "note: WANDB_API_KEY set but WANDB_ENTITY unset -> console logging only (set WANDB_ENTITY to enable W&B)"
  LOGGER='[console]'
fi

[[ -f "${TRAIN_PARQUET}" ]] || { echo "missing TRAIN_PARQUET: ${TRAIN_PARQUET}" >&2; exit 1; }
mkdir -p "${SAVE_DIR}"

echo "============================================================"
echo "model:       ${MODEL_PATH}"
echo "train data:  ${TRAIN_PARQUET}"
echo "checkpoints: ${SAVE_DIR}"
echo "GPUs:        ${NPROC}   ulysses_sp: ${ULYSSES_SP_SIZE}"
echo "max_length:  ${MAX_LENGTH}   global bs: ${TRAIN_BATCH_SIZE}   epochs: ${TOTAL_EPOCHS}"
echo "logger:      ${LOGGER}"
echo "============================================================"

torchrun --nnodes=1 --nproc_per_node="${NPROC}" \
    -m verl.trainer.sft_trainer \
        data.train_files="${TRAIN_PARQUET}" \
        data.val_files=null \
        data.messages_key=messages \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.micro_batch_size_per_gpu="${MICRO_BATCH_PER_GPU}" \
        data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
        data.use_dynamic_bsz=true \
        data.max_length="${MAX_LENGTH}" \
        data.truncation=right \
        data.pad_mode=no_padding \
        data.ignore_input_ids_mismatch=true \
        model.path="${MODEL_PATH}" \
        model.use_remove_padding=true \
        model.enable_gradient_checkpointing=true \
        model.enable_activation_offload=true \
        engine.strategy=fsdp \
        engine.dtype=bfloat16 \
        engine.param_offload=true \
        engine.optimizer_offload=true \
        engine.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
        optim.lr="${LR}" \
        optim.weight_decay="${WEIGHT_DECAY}" \
        optim.lr_warmup_steps_ratio="${WARMUP_RATIO}" \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.test_freq=-1 \
        trainer.logger="${LOGGER}" \
        trainer.experiment_name="$(basename "${SAVE_DIR}")" \
        trainer.default_local_dir="${SAVE_DIR}" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node="${NPROC}" \
        trainer.resume_mode=auto \
        ${TOTAL_TRAINING_STEPS:+trainer.total_training_steps=${TOTAL_TRAINING_STEPS}}

echo "done. checkpoints -> ${SAVE_DIR}"
