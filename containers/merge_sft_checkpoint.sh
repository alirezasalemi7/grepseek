#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
export CONTAINER_DISABLE_GPU=1
container_init

SFT_CKPT_DIR="${SFT_CKPT_DIR:-${CKPT_DIR:-}}"
: "${SFT_CKPT_DIR:?set SFT_CKPT_DIR to an SFT FSDP checkpoint directory, e.g. output/sft/global_step_200}"

SFT_CKPT_HOST="$(container_abs_path "${SFT_CKPT_DIR}")"
container_require_dir "${SFT_CKPT_HOST}"

HF_DIR_HOST="$(container_abs_path "${HF_DIR:-${SFT_CKPT_HOST}/huggingface}")"

container_add_existing_path_bind_same "${SFT_CKPT_HOST}"
container_add_parent_bind_same "${HF_DIR_HOST}"
container_add_env SFT_CKPT_DIR "${SFT_CKPT_HOST}"
container_add_env HF_DIR "${HF_DIR_HOST}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"
container_add_env ACCELERATE_USE_DEEPSPEED false
container_add_env CUDA_VISIBLE_DEVICES ""

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
export PYTHONPATH=/workspace/verl:${PYTHONPATH:-}
cd /workspace

echo "Merging SFT FSDP checkpoint: ${SFT_CKPT_DIR} -> ${HF_DIR}"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${SFT_CKPT_DIR}" \
  --target_dir "${HF_DIR}" \
  --trust-remote-code \
  "$@"
' _ "$@"
