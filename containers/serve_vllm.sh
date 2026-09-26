#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

: "${MODEL_PATH:?set MODEL_PATH to a HuggingFace model ID or local HF checkpoint directory}"

PORT="${PORT:-10730}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-grepseek}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.90}"

if [[ "${MODEL_PATH}" = /* && -d "${MODEL_PATH}" ]]; then
  container_add_existing_path_bind_same "${MODEL_PATH}"
fi

container_add_port "${PORT}"
container_add_env MODEL_PATH "${MODEL_PATH}"
container_add_env PORT "${PORT}"
container_add_env HOST "${HOST}"
container_add_env TP_SIZE "${TP_SIZE}"
container_add_env SERVED_MODEL_NAME "${SERVED_MODEL_NAME}"
container_add_env MAX_MODEL_LEN "${MAX_MODEL_LEN}"
container_add_env MAX_NUM_SEQS "${MAX_NUM_SEQS}"
container_add_env GPU_UTIL "${GPU_UTIL}"
container_add_env ENABLE_AUTO_TOOL_CHOICE "${ENABLE_AUTO_TOOL_CHOICE:-1}"
container_add_env TOOL_CALL_PARSER "${TOOL_CALL_PARSER:-qwen3_coder}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
exec bash rl/serve_rl.sh
'
