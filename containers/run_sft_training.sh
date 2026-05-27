#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

: "${TRAIN_PARQUET:?set TRAIN_PARQUET to the SFT train.parquet path}"

TRAIN_PARQUET_HOST="$(container_abs_path "${TRAIN_PARQUET}")"
container_require_file "${TRAIN_PARQUET_HOST}"
container_add_existing_path_bind_same "${TRAIN_PARQUET_HOST}"
container_add_env TRAIN_PARQUET "${TRAIN_PARQUET_HOST}"

if [[ "${MODEL_PATH:-}" = /* && -d "${MODEL_PATH}" ]]; then
  container_add_existing_path_bind_same "${MODEL_PATH}"
fi
if [[ -n "${SAVE_DIR:-}" ]]; then
  container_add_parent_bind_same "${SAVE_DIR}"
fi

container_add_env MODEL_PATH "${MODEL_PATH:-Qwen/Qwen3.5-9B}"
container_add_env NPROC "${NPROC:-}"
container_add_env SAVE_DIR "${SAVE_DIR:-}"
container_add_env MAX_LENGTH "${MAX_LENGTH:-16384}"
container_add_env TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE:-32}"
container_add_env MICRO_BATCH_PER_GPU "${MICRO_BATCH_PER_GPU:-1}"
container_add_env MAX_TOKEN_LEN_PER_GPU "${MAX_TOKEN_LEN_PER_GPU:-16384}"
container_add_env ULYSSES_SP_SIZE "${ULYSSES_SP_SIZE:-}"
container_add_env LR "${LR:-5e-6}"
container_add_env WEIGHT_DECAY "${WEIGHT_DECAY:-0.01}"
container_add_env WARMUP_RATIO "${WARMUP_RATIO:-0.05}"
container_add_env TOTAL_EPOCHS "${TOTAL_EPOCHS:-1}"
container_add_env SAVE_FREQ "${SAVE_FREQ:-50}"
container_add_env TOTAL_TRAINING_STEPS "${TOTAL_TRAINING_STEPS:-}"
container_add_env WANDB_API_KEY "${WANDB_API_KEY:-}"
container_add_env WANDB_ENTITY "${WANDB_ENTITY:-}"
container_add_env WANDB_PROJECT "${WANDB_PROJECT:-}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
exec bash sft/run_sft.sh "$@"
' _ "$@"
