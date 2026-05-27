#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

: "${BASE_URL:?set BASE_URL to the OpenAI-compatible vLLM endpoint, e.g. http://HOST:PORT/v1}"
: "${GREPSEEK_CORPUS_ROOT:?set GREPSEEK_CORPUS_ROOT to the directory containing wiki_corpus.jsonl}"

CORPUS_HOST="$(container_abs_path "${GREPSEEK_CORPUS_ROOT}")"
container_require_dir "${CORPUS_HOST}"
container_require_file "${CORPUS_HOST}/wiki_corpus.jsonl"

MODEL="${MODEL:-grepseek}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
API_KEY="${API_KEY:-EMPTY}"

container_add_bind "${CORPUS_HOST}" /corpus
container_add_env BASE_URL "${BASE_URL}"
container_add_env MODEL "${MODEL}"
container_add_env TOKENIZER "${TOKENIZER}"
container_add_env API_KEY "${API_KEY}"
container_add_env GREPSEEK_CORPUS_ROOT /corpus
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
exec bash inference/run_inference.sh \
  --base_url "${BASE_URL}" \
  --api_key "${API_KEY}" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  "$@"
' _ "$@"
