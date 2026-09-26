#!/usr/bin/env bash
# Merge a verl RL (GRPO) FSDP checkpoint into a single HuggingFace-format dir so
# it can be served / evaluated. The RL trainer nests the actor shards under an
# `actor/` subdir (GRPO has no critic), so the merger runs against CKPT_DIR/actor.
#
# Input:  CKPT_DIR  — a verl RL save dir (.../global_step_NNN) containing
#                     actor/model_world_size_<W>_rank_<R>.pt shards + actor/huggingface/
#                     (tokenizer + config).
# Output: HF_DIR    — defaults to CKPT_DIR/actor/huggingface (weights written
#                     alongside the existing tokenizer/config).
#
# Usage:
#   CKPT_DIR=/path/to/checkpoints/rl/<run>/global_step_200 bash rl/convert_to_hf.sh
#   CKPT_DIR=... HF_DIR=/path/to/dest bash rl/convert_to_hf.sh
set -euo pipefail
REPO_ROOT="${PWD}"
if [[ ! -f "${REPO_ROOT}/README.md" || ! -d "${REPO_ROOT}/sft" || ! -d "${REPO_ROOT}/rl" || ! -d "${REPO_ROOT}/inference" ]]; then
  echo "error: run this command from the grepseek repo root, e.g.:" >&2
  echo "       CKPT_DIR=checkpoints/rl/<run>/global_step_200 bash rl/convert_to_hf.sh" >&2
  exit 2
fi
root_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}
export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to a verl RL save dir, e.g. .../checkpoints/rl/<run>/global_step_200}"
CKPT_DIR="$(root_path "${CKPT_DIR}")"
HF_DIR="${HF_DIR:-${CKPT_DIR}/actor/huggingface}"
HF_DIR="$(root_path "${HF_DIR}")"

[[ -d "${CKPT_DIR}/actor" ]] || { echo "missing ${CKPT_DIR}/actor (RL ckpt layout expects FSDP shards under actor/)" >&2; exit 1; }
[[ -d "${CKPT_DIR}/actor/huggingface" ]] || { echo "missing ${CKPT_DIR}/actor/huggingface (config.json + tokenizer)" >&2; exit 1; }
shopt -s nullglob
shards=( "${CKPT_DIR}"/actor/model_world_size_*_rank_*.pt )
shopt -u nullglob
[[ ${#shards[@]} -gt 0 ]] || { echo "no model_world_size_*_rank_*.pt shards in ${CKPT_DIR}/actor" >&2; exit 1; }

echo "merging ${#shards[@]} FSDP shards: ${CKPT_DIR}/actor -> ${HF_DIR}"
mkdir -p "${HF_DIR}"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/actor" \
    --target_dir "${HF_DIR}" \
    --trust-remote-code

echo "done. HF checkpoint -> ${HF_DIR}"
