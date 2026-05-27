#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

: "${LLM_MODEL:?set LLM_MODEL to your served teacher model name}"
: "${CORPUS_DIR:?set CORPUS_DIR to the directory containing wiki_corpus.jsonl}"

CORPUS_HOST="$(container_abs_path "${CORPUS_DIR}")"
container_require_dir "${CORPUS_HOST}"
container_require_file "${CORPUS_HOST}/wiki_corpus.jsonl"

RUN_NAME="${RUN_NAME:-container_$(date +%Y%m%d_%H%M%S)}"
DATASETS="${DATASETS:-hotpotqa,nq}"
SPLIT="${SPLIT:-train}"
HOTPOT_N="${HOTPOT_N:-100}"
NQ_N="${NQ_N:-100}"
HOTPOT_START="${HOTPOT_START:-${START:-0}}"
NQ_START="${NQ_START:-${START:-0}}"
PARALLEL_EXAMPLES="${PARALLEL_EXAMPLES:-8}"
BUILD_PARQUET="${BUILD_PARQUET:-1}"
MAX_TOOL_CALLS="${MAX_TOOL_CALLS:-10}"
BACKWARD_MAX_ITERATIONS="${BACKWARD_MAX_ITERATIONS:-6}"

container_add_bind "${CORPUS_HOST}" /corpus
container_add_env CORPUS_DIR /corpus
container_add_env LLM_MODEL "${LLM_MODEL}"
container_add_env LLM_HOST "${LLM_HOST:-127.0.0.1}"
container_add_env LLM_PORT "${LLM_PORT:-8000}"
container_add_env DATASETS "${DATASETS}"
container_add_env SPLIT "${SPLIT}"
container_add_env HOTPOT_N "${HOTPOT_N}"
container_add_env NQ_N "${NQ_N}"
container_add_env HOTPOT_START "${HOTPOT_START}"
container_add_env NQ_START "${NQ_START}"
container_add_env PARALLEL_EXAMPLES "${PARALLEL_EXAMPLES}"
container_add_env BUILD_PARQUET "${BUILD_PARQUET}"
container_add_env RUN_NAME "${RUN_NAME}"
container_add_env MAX_TOOL_CALLS "${MAX_TOOL_CALLS}"
container_add_env BACKWARD_MAX_ITERATIONS "${BACKWARD_MAX_ITERATIONS}"
container_add_env OPENAI_API_KEY "${OPENAI_API_KEY:-}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace

OUT_ROOT="sft/data_generation/output/${RUN_NAME}"
mkdir -p "${OUT_ROOT}"

IFS="," read -r -a datasets <<< "${DATASETS}"
for dataset in "${datasets[@]}"; do
  dataset="$(echo "${dataset}" | tr -d "[:space:]")"
  [[ -n "${dataset}" ]] || continue
  case "${dataset}" in
    hotpotqa)
      n="${HOTPOT_N}"
      start="${HOTPOT_START}"
      ;;
    nq)
      n="${NQ_N}"
      start="${NQ_START}"
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 2
      ;;
  esac

  dataset_dir="${OUT_ROOT}/${dataset}"
  mkdir -p "${dataset_dir}"
  echo "Collecting ${dataset}: n=${n}, start=${start}, split=${SPLIT}"
  python sft/data_generation/create_data.py \
    --dataset "${dataset}" \
    --split "${SPLIT}" \
    --n "${n}" \
    --start "${start}" \
    --corpus_dir "${CORPUS_DIR}" \
    --out "${dataset_dir}/traces.jsonl" \
    --out_chatml "${dataset_dir}/sft.jsonl" \
    --out_pretty "${dataset_dir}/pretty.txt" \
    --parallel_examples "${PARALLEL_EXAMPLES}" \
    --max_tool_calls "${MAX_TOOL_CALLS}" \
    --backward_max_iterations "${BACKWARD_MAX_ITERATIONS}" \
    "$@"
done

MERGED="${OUT_ROOT}/sft_merged.jsonl"
: > "${MERGED}"
for f in "${OUT_ROOT}"/*/sft.jsonl; do
  [[ -s "${f}" ]] && cat "${f}" >> "${MERGED}"
done

if [[ ! -s "${MERGED}" ]]; then
  echo "No successful SFT records were written to ${MERGED}" >&2
  exit 1
fi

echo "Merged SFT jsonl: ${MERGED}"
if [[ "${BUILD_PARQUET}" == "1" || "${BUILD_PARQUET,,}" == "true" ]]; then
  python sft/data_generation/to_parquet.py \
    --in "${MERGED}" \
    --out_dir "${OUT_ROOT}/sft_parquet" \
    --include_tools
  echo "Parquet output: ${OUT_ROOT}/sft_parquet"
fi
' _ "$@"
