#!/usr/bin/env bash
# Serve a trained GrepSeek checkpoint with vLLM (OpenAI-compatible API) for
# evaluation / interactive use. MODEL_PATH must be an HF-format dir — i.e. the
# output of convert_to_hf.sh (a verl RL save dir merged to HuggingFace format),
# or a merged SFT checkpoint.
#
# Usage:
#   MODEL_PATH=/path/to/global_step_200/actor/huggingface bash serve_rl.sh
#   MODEL_PATH=... PORT=10730 TP_SIZE=2 bash serve_rl.sh
#
# The serve flags mirror the paper's eval server: Qwen3 reasoning parser, native
# tool calling (so the model can emit <tool_call> blocks), 16K context.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to an HF checkpoint dir (output of convert_to_hf.sh)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-grepseek}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10730}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.90}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

[[ -d "${MODEL_PATH}" ]] || { echo "missing MODEL_PATH: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "${MODEL_PATH} has no config.json — not an HF checkpoint?" >&2; exit 1; }
if ! ls "${MODEL_PATH}"/model*.safetensors >/dev/null 2>&1 \
   && ! ls "${MODEL_PATH}"/pytorch_model*.bin >/dev/null 2>&1; then
  echo "${MODEL_PATH} has no model weights — run convert_to_hf.sh first." >&2
  exit 1
fi

echo "serving ${SERVED_MODEL_NAME} from ${MODEL_PATH}"
echo "  http://${HOST}:${PORT}/v1   (TP=${TP_SIZE}, max_model_len=${MAX_MODEL_LEN})"

vllm_cmd=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --reasoning-parser qwen3
  --language-model-only
  --trust-remote-code
)
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" || "${ENABLE_AUTO_TOOL_CHOICE,,}" == "true" ]]; then
  vllm_cmd+=(--enable-auto-tool-choice)
  [[ -n "${TOOL_CALL_PARSER}" ]] && vllm_cmd+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi

echo "Command: ${vllm_cmd[*]}"
exec "${vllm_cmd[@]}"
