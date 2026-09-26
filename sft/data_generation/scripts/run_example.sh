#!/usr/bin/env bash
# Minimal end-to-end example: generate 10 HotpotQA cold-start trajectories and
# convert them to train/val parquet.
#
# Prerequisites (see README.md):
#   - pip install -r sft/data_generation/requirements.txt   and   ripgrep (`rg`) on PATH
#   - an OpenAI-compatible teacher server reachable at $LLM_HOST:$LLM_PORT
#     serving model $LLM_MODEL
#   - the wiki-18 corpus at data/wiki_18_corpus/wiki_corpus.jsonl, or set
#     $CORPUS_DIR to a directory containing wiki_corpus.jsonl
set -euo pipefail
REPO_ROOT="${PWD}"
if [[ ! -f "${REPO_ROOT}/README.md" || ! -d "${REPO_ROOT}/sft/data_generation" ]]; then
  echo "error: run this command from the grepseek repo root, e.g.:" >&2
  echo "       bash sft/data_generation/scripts/run_example.sh" >&2
  exit 2
fi

: "${LLM_MODEL:?set LLM_MODEL to your served model name (vLLM --served-model-name)}"
export CORPUS_DIR="${CORPUS_DIR:-${REPO_ROOT}/data/wiki_18_corpus}"
export LLM_HOST="${LLM_HOST:-127.0.0.1}"
export LLM_PORT="${LLM_PORT:-8000}"

if [[ ! -f "${CORPUS_DIR}/wiki_corpus.jsonl" ]]; then
  echo "error: ${CORPUS_DIR}/wiki_corpus.jsonl not found." >&2
  echo "       Download it with: python sft/data_generation/download_corpus.py --dest data/wiki_18_corpus" >&2
  echo "       Or set CORPUS_DIR to a directory containing wiki_corpus.jsonl." >&2
  exit 1
fi

python sft/data_generation/create_data.py \
  --dataset hotpotqa --split train --n 10 \
  --corpus_dir "$CORPUS_DIR" \
  --out sft/data_generation/output/traces.jsonl \
  --out_chatml sft/data_generation/output/sft.jsonl \
  --out_pretty sft/data_generation/output/pretty.txt \
  --parallel_examples 4

# Turn the successful trajectories into train/val parquet for SFT:
python sft/data_generation/to_parquet.py \
  --in 'sft/data_generation/output/sft.jsonl' \
  --out_dir sft/data_generation/output/sft_parquet \
  --include_tools

echo
echo "Done. Inspect sft/data_generation/output/pretty.txt for human-readable trajectories,"
echo "sft/data_generation/output/sft.jsonl for the SFT messages, and"
echo "sft/data_generation/output/sft_parquet/ for training files."
