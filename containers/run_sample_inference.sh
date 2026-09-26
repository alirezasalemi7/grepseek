#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

: "${BASE_URL:?set BASE_URL to the OpenAI-compatible vLLM endpoint, e.g. http://HOST:PORT/v1}"

MODEL="${MODEL:-grepseek}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3.5-9B}"
API_KEY="${API_KEY:-EMPTY}"
SAMPLE_DIR="${CACHE_DIR}/sample_inference"
mkdir -p "${SAMPLE_DIR}/corpus" "${SAMPLE_DIR}/out"

cat > "${SAMPLE_DIR}/corpus/wiki_corpus.jsonl" <<'EOF'
{"id":"smoke-doc-1","contents":"The answer to the GrepSeek container sample test is bluefin. This line exists only for container inference validation."}
{"id":"smoke-doc-2","contents":"A distractor passage says the answer is not marigold and not basalt."}
EOF

cat > "${SAMPLE_DIR}/questions.jsonl" <<'EOF'
{"id":"sample-question-1","question":"What is the answer to the GrepSeek container sample test?","golden_answers":["bluefin"]}
EOF

container_add_env BASE_URL "${BASE_URL}"
container_add_env MODEL "${MODEL}"
container_add_env TOKENIZER "${TOKENIZER}"
container_add_env API_KEY "${API_KEY}"
container_add_env GREPSEEK_CORPUS_ROOT /cache/sample_inference/corpus
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
bash inference/run_inference.sh \
  --base_url "${BASE_URL}" \
  --api_key "${API_KEY}" \
  --model "${MODEL}" \
  --tokenizer "${TOKENIZER}" \
  --input /cache/sample_inference/questions.jsonl \
  --out_dir /cache/sample_inference/out \
  --parallel 1 \
  --limit 1 \
  --max_assistant_turns "${MAX_ASSISTANT_TURNS:-3}" \
  --max_tokens_per_turn "${MAX_TOKENS_PER_TURN:-512}" \
  --tool_timeout "${TOOL_TIMEOUT:-20}" \
  --no_eval \
  "$@"

python - <<'"'"'PY'"'"'
import json
from pathlib import Path

predictions = Path("/cache/sample_inference/out/questions/predictions.jsonl")
if not predictions.is_file() or predictions.stat().st_size == 0:
    raise SystemExit(f"missing or empty predictions file: {predictions}")

rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"expected one prediction row, got {len(rows)}")

row = rows[0]
errors = []
for turn in row.get("turns", []):
    if turn.get("role") != "tool":
        continue
    payload = json.loads(turn.get("content") or "{}")
    stderr = (payload.get("stderr") or "").strip()
    if stderr or payload.get("timed_out") or payload.get("exit_code") not in (0, None):
        errors.append({"command": payload.get("command"), "stderr": stderr, "exit_code": payload.get("exit_code")})

if errors:
    raise SystemExit(json.dumps(errors, indent=2))
if not row.get("prediction"):
    raise SystemExit("sample inference produced no final prediction")

print(f"Sample prediction: {row['prediction']}")
PY
' _ "$@"
