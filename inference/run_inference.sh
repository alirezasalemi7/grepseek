#!/usr/bin/env bash
# Thin launcher for the GrepSeek inference harness (inference.run). Sets up
# PYTHONPATH + caches and forwards all args to `python -m inference.run`.
#
# Generation (no benchmark, gold optional):
#   GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus \
#     bash inference/run_inference.sh --base_url http://HOST:PORT/v1 --model grepseek \
#       --input my_questions.jsonl --out_dir output/gen
#
# Benchmark eval (Search-R1 QA suite, scored EM/F1):
#   GREPSEEK_CORPUS_ROOT=/path/to/wiki_18_corpus \
#     bash inference/run_inference.sh --base_url http://HOST:PORT/v1 --model grepseek \
#       --datasets nq,hotpotqa,bamboogle --limit 200 --out_dir output/eval
#
# Serve the model first with rl/serve_rl.sh (or any OpenAI-compatible vLLM
# server). Run `python -m inference.run --help` for the full flag list.
set -euo pipefail
REPO_ROOT="${PWD}"                   # run launchers from the repo root
if [[ ! -f "${REPO_ROOT}/README.md" || ! -d "${REPO_ROOT}/sft" || ! -d "${REPO_ROOT}/rl" || ! -d "${REPO_ROOT}/inference" ]]; then
  echo "error: run this command from the grepseek repo root, e.g.:" >&2
  echo "       bash inference/run_inference.sh --help" >&2
  exit 2
fi
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# caches off the home directory
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HUB_CACHE}"

# Corpus dir: env wins; otherwise use the repo-local default. run.py also
# accepts --corpus_dir, which overrides this argparse default.
export GREPSEEK_CORPUS_ROOT="${GREPSEEK_CORPUS_ROOT:-${REPO_ROOT}/data/wiki_18_corpus}"

exec python -m inference.run "$@"
