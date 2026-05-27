#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

: "${GREPSEEK_TRAIN_FILES:?set GREPSEEK_TRAIN_FILES to the RL train JSONL path}"
: "${GREPSEEK_VAL_FILES:?set GREPSEEK_VAL_FILES to the RL val/dev JSONL path}"
: "${GREPSEEK_CORPUS_ROOT:?set GREPSEEK_CORPUS_ROOT to the directory containing wiki_corpus.jsonl}"

TRAIN_HOST="$(container_abs_path "${GREPSEEK_TRAIN_FILES}")"
VAL_HOST="$(container_abs_path "${GREPSEEK_VAL_FILES}")"
CORPUS_HOST="$(container_abs_path "${GREPSEEK_CORPUS_ROOT}")"
container_require_file "${TRAIN_HOST}"
container_require_file "${VAL_HOST}"
container_require_dir "${CORPUS_HOST}"
container_require_file "${CORPUS_HOST}/wiki_corpus.jsonl"

container_add_existing_path_bind_same "${TRAIN_HOST}"
container_add_existing_path_bind_same "${VAL_HOST}"
container_add_bind "${CORPUS_HOST}" /corpus

if [[ "${GREPSEEK_MODEL_PATH:-}" = /* && -d "${GREPSEEK_MODEL_PATH}" ]]; then
  container_add_existing_path_bind_same "${GREPSEEK_MODEL_PATH}"
fi
if [[ -n "${GREPSEEK_OUTPUT_DIR:-}" ]]; then
  container_add_parent_bind_same "${GREPSEEK_OUTPUT_DIR}"
fi

container_add_env GREPSEEK_TRAIN_FILES "${TRAIN_HOST}"
container_add_env GREPSEEK_VAL_FILES "${VAL_HOST}"
container_add_env GREPSEEK_CORPUS_ROOT /corpus
container_add_env GREPSEEK_MODEL_PATH "${GREPSEEK_MODEL_PATH:-Qwen/Qwen3.5-9B}"
container_add_env GREPSEEK_OUTPUT_DIR "${GREPSEEK_OUTPUT_DIR:-}"
container_add_env NPROC "${NPROC:-}"
container_add_env ULYSSES_SP "${ULYSSES_SP:-2}"
container_add_env WANDB_API_KEY "${WANDB_API_KEY:-}"
container_add_env WANDB_ENTITY "${WANDB_ENTITY:-}"
container_add_env WANDB_PROJECT "${WANDB_PROJECT:-}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
exec bash rl/run_rl.sh "$@"
' _ "$@"
