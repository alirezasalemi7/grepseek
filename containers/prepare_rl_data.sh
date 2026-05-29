#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/container_runtime.sh"
container_init

OUT_DIR_HOST="$(container_abs_path "${OUT_DIR:-data/rl/nq_hotpot}")"
container_add_parent_bind_same "${OUT_DIR_HOST}"
container_add_env OUT_DIR "${OUT_DIR_HOST}"
container_add_env HF_TOKEN "${HF_TOKEN:-}"

container_exec grepseek bash -lc '
set -euo pipefail
export PATH=/opt/envs/grepseek/bin:${PATH}
cd /workspace
python rl/prepare_rl_data.py --out_dir "${OUT_DIR}" "$@"
' _ "$@"
