#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

REGISTRY="${REGISTRY:-ghcr.io/alirezasalemi7}"
TAG="${TAG:-v1}"
IMAGE_DIR="${IMAGE_DIR:-${SCRIPT_DIR}/images}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SCRIPT_DIR}/apptainer_cache}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SCRIPT_DIR}/apptainer_tmp}"
INCLUDE_ALL=0

usage() {
  cat <<'USAGE'
Usage:
  bash <project root>/containers/pull_images.sh [--all]

Defaults:
  Pulls grepseek and grepseek-retriever.
  Does not pull grepseek-all unless --all is passed.

Options:
  --all                Also pull ghcr.io/alirezasalemi7/grepseek-all
  --tag TAG            Image tag to pull. Default: v1
  --registry REGISTRY  Image registry/namespace. Default: ghcr.io/alirezasalemi7
  --image-dir PATH     Absolute output directory for SIF files.
  -h, --help           Show this help.

For private GHCR packages, set:
  export APPTAINER_DOCKER_USERNAME=<github-user>
  export APPTAINER_DOCKER_PASSWORD=<github-token-with-read:packages>
USAGE
}

require_absolute_path() {
  local name="$1"
  local path="$2"
  case "${path}" in
    /*) ;;
    *)
      echo "ERROR: ${name} must be an absolute path, got: ${path}" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      INCLUDE_ALL=1
      shift
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_absolute_path PROJECT_ROOT "${PROJECT_ROOT}"
require_absolute_path IMAGE_DIR "${IMAGE_DIR}"
require_absolute_path APPTAINER_CACHEDIR "${APPTAINER_CACHEDIR}"
require_absolute_path APPTAINER_TMPDIR "${APPTAINER_TMPDIR}"

mkdir -p "${IMAGE_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

export APPTAINER_CACHEDIR
export APPTAINER_TMPDIR
export TMPDIR="${APPTAINER_TMPDIR}"

tag_for_file="${TAG//\//_}"
tag_for_file="${tag_for_file//:/_}"

pull_one() {
  local image_name="$1"
  local sif_path="${IMAGE_DIR}/${image_name}_${tag_for_file}.sif"
  require_absolute_path SIF_PATH "${sif_path}"

  echo
  echo "Pulling ${REGISTRY}/${image_name}:${TAG}"
  echo "Writing ${sif_path}"
  apptainer pull --force "${sif_path}" "docker://${REGISTRY}/${image_name}:${TAG}"
}

pull_one grepseek
pull_one grepseek-retriever

if [[ "${INCLUDE_ALL}" == "1" ]]; then
  pull_one grepseek-all
fi
