#!/usr/bin/env bash
# GRPO RL training for GrepSeek via verl's `main_ppo` (verl is the engine; the
# agent loop, search/grep tool, reward, and dataset are GrepSeek's, registered
# through the configs under grepseek/trainer/config). Reads the config at
# grepseek/trainer/config/grpo_trainer.yaml — the paper's "ours" GRPO run.
#
# Quick start (4×A100-80GB):
#   export GREPSEEK_MODEL_PATH=/path/to/sft_checkpoint/global_step_NNN/huggingface
#   export GREPSEEK_TRAIN_FILES=/path/to/rl/train.jsonl
#   export GREPSEEK_VAL_FILES=/path/to/rl/dev.jsonl
#   export GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus      # dir holding wiki_corpus.jsonl
#   export NPROC=4
#   bash rl/run_rl.sh
#
# All knobs are environment variables (see below). Extra args are passed through
# to main_ppo, e.g.:  bash rl/run_rl.sh actor_rollout_ref.actor.optim.lr=2e-6
# Runs on any machine with >=2 GPUs once your training env is active.
set -euo pipefail
REPO_ROOT="${PWD}"                   # run launchers from the repo root
if [[ ! -f "${REPO_ROOT}/README.md" || ! -d "${REPO_ROOT}/sft" || ! -d "${REPO_ROOT}/rl" || ! -d "${REPO_ROOT}/inference" ]]; then
  echo "error: run this command from the grepseek repo root, e.g.:" >&2
  echo "       bash rl/run_rl.sh" >&2
  exit 2
fi
root_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}
export GREPSEEK_ROOT="${REPO_ROOT}"

# --- verl + grepseek on PYTHONPATH (both used via PYTHONPATH, not pip-installed) ---
VERL_DIR="${VERL_DIR:-${REPO_ROOT}/verl}"
if [[ ! -d "${VERL_DIR}/verl/trainer" ]]; then
  echo "error: vendored verl not found at ${VERL_DIR} (expected ${VERL_DIR}/verl/trainer)" >&2
  exit 1
fi
export PYTHONPATH="${REPO_ROOT}/rl:${VERL_DIR}:${PYTHONPATH:-}"

# --- caches (keep off the home directory) ---
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"
# Triton autotuner races across Ray/torchrun workers on shared filesystems; pin
# it to local scratch (override TRITON_CACHE_DIR for node-local disk).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_$$}"
mkdir -p "${TRITON_CACHE_DIR}"
# CUDA allocator: do NOT set expandable_segments for RL. vLLM's CuMemAllocator
# (the hybrid-engine memory pool that swaps weights between FSDP-train and vLLM
# rollout) is incompatible with it and crashes the rollout engine with
# "Expandable segments are not compatible with memory pool" (pytorch #147851).
# Leave the allocator at its default. (The SFT launcher DOES set it — SFT has no
# vLLM in the loop — but RL must not.)
export TOKENIZERS_PARALLELISM=true

# --- required paths ---
: "${GREPSEEK_TRAIN_FILES:?set GREPSEEK_TRAIN_FILES to the RL train JSONL}"
: "${GREPSEEK_VAL_FILES:?set GREPSEEK_VAL_FILES to the RL val/dev JSONL}"
GREPSEEK_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT:-${REPO_ROOT}/data/wiki_18_corpus}"
GREPSEEK_TRAIN_FILES="$(root_path "${GREPSEEK_TRAIN_FILES}")"
GREPSEEK_VAL_FILES="$(root_path "${GREPSEEK_VAL_FILES}")"
GREPSEEK_CORPUS_ROOT="$(root_path "${GREPSEEK_CORPUS_ROOT}")"
export GREPSEEK_TRAIN_FILES GREPSEEK_VAL_FILES
# The search tool reads VERL_TOOL_CORPUS_ROOT directly (takes precedence over the
# yaml corpus_root). It expects ${GREPSEEK_CORPUS_ROOT}/wiki_corpus.jsonl to exist.
export VERL_TOOL_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT}"
if [[ ! -f "${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl" ]]; then
  echo "error: ${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl not found." >&2
  echo "       Download it with sft/data_generation/download_corpus.py (see README)." >&2
  exit 1
fi

# --- model / output / GPUs ---
export GREPSEEK_MODEL_PATH="${GREPSEEK_MODEL_PATH:-Qwen/Qwen3.5-9B}"
[[ "${GREPSEEK_MODEL_PATH}" == "Qwen/Qwen3.5-9B" ]] && \
  echo "note: GREPSEEK_MODEL_PATH unset -> initializing RL from the BASE model. The paper inits from the cold-start SFT checkpoint."
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NPROC="${NPROC:-4}"
ULYSSES_SP="${ULYSSES_SP:-2}"   # must divide NPROC; paper used 2 on 4 GPUs
export GREPSEEK_OUTPUT_DIR="${GREPSEEK_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/rl/$(date +%Y%m%d-%H%M%S)}"
GREPSEEK_OUTPUT_DIR="$(root_path "${GREPSEEK_OUTPUT_DIR}")"
export GREPSEEK_OUTPUT_DIR
mkdir -p "${GREPSEEK_OUTPUT_DIR}"

# --- optional Weights & Biases (console-only unless BOTH a key and entity are set) ---
if [[ -n "${WANDB_API_KEY:-}" && -n "${WANDB_ENTITY:-}" ]]; then
  LOGGER='[console,wandb]'; WANDB_ENABLE=true
  export WANDB_PROJECT="${WANDB_PROJECT:-grepseek_verl_grpo}"
else
  [[ -n "${WANDB_API_KEY:-}" ]] && echo "note: WANDB_API_KEY set but WANDB_ENTITY unset -> console logging only"
  LOGGER='[console]'; WANDB_ENABLE=false
fi

# --- register the agent loop (fail fast if the package isn't importable) ---
python -c "import grepseek.trainer; print('GrepSeek agent loop registered.')"

echo "============================================================"
echo "model:       ${GREPSEEK_MODEL_PATH}"
echo "train data:  ${GREPSEEK_TRAIN_FILES}"
echo "val data:    ${GREPSEEK_VAL_FILES}"
echo "corpus:      ${VERL_TOOL_CORPUS_ROOT}/wiki_corpus.jsonl"
echo "output:      ${GREPSEEK_OUTPUT_DIR}"
echo "GPUs:        ${NPROC}   ulysses_sp: ${ULYSSES_SP}"
echo "logger:      ${LOGGER}"
echo "============================================================"

python -m verl.trainer.main_ppo \
    --config-path "${REPO_ROOT}/rl/grepseek/trainer/config" \
    --config-name grpo_trainer \
    "hydra.searchpath=[file://${VERL_DIR}/verl/trainer/config]" \
    trainer.n_gpus_per_node="${NPROC}" \
    ray_kwargs.ray_init.num_gpus="${NPROC}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${NPROC}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="${ULYSSES_SP}" \
    trainer.logger="${LOGGER}" \
    wandb.enable="${WANDB_ENABLE}" \
    "$@"

echo "done. checkpoints -> ${GREPSEEK_OUTPUT_DIR}"
