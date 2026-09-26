#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
export CONTAINER_DISABLE_GPU=1
container_init

: "${CKPT_DIR:?set CKPT_DIR to an RL checkpoint directory, e.g. output/rl/global_step_200}"

CKPT_HOST="$(container_abs_path "${CKPT_DIR}")"
container_require_dir "${CKPT_HOST}"

container_add_existing_path_bind_same "${CKPT_HOST}"
container_add_env CKPT_DIR "${CKPT_HOST}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"
container_add_env ACCELERATE_USE_DEEPSPEED false
container_add_env CUDA_VISIBLE_DEVICES ""

if [[ -n "${HF_DIR:-}" ]]; then
  HF_DIR_HOST="$(container_abs_path "${HF_DIR}")"
  container_add_parent_bind_same "${HF_DIR_HOST}"
  container_add_env HF_DIR "${HF_DIR_HOST}"
fi

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
exec bash rl/convert_to_hf.sh
'
