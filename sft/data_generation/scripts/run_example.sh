#!/usr/bin/env bash
# Minimal end-to-end example: generate 10 HotpotQA cold-start trajectories and
# convert them to train/val parquet.
#
# Prerequisites (see README.md):
#   - pip install -r requirements.txt   and   ripgrep (`rg`) on PATH
#   - an OpenAI-compatible teacher server reachable at $LLM_HOST:$LLM_PORT
#     serving model $LLM_MODEL
#   - the wiki-18 corpus: $CORPUS_DIR must contain wiki_corpus.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."   # run from the package root

: "${LLM_MODEL:?set LLM_MODEL to your served model name (vLLM --served-model-name)}"
: "${CORPUS_DIR:?set CORPUS_DIR to a directory containing wiki_corpus.jsonl}"
export LLM_HOST="${LLM_HOST:-127.0.0.1}"
export LLM_PORT="${LLM_PORT:-8000}"

python create_data.py \
  --dataset hotpotqa --split train --n 10 \
  --corpus_dir "$CORPUS_DIR" \
  --out output/traces.jsonl \
  --out_chatml output/sft.jsonl \
  --out_pretty output/pretty.txt \
  --parallel_examples 4

# Turn the successful trajectories into train/val parquet for SFT:
python to_parquet.py --in 'output/sft.jsonl' --out_dir output/sft_parquet --include_tools

echo
echo "Done. Inspect output/pretty.txt for human-readable trajectories,"
echo "output/sft.jsonl for the SFT messages, and output/sft_parquet/ for training files."
