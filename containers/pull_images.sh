#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PWD}"
if [[ ! -f "${PROJECT_ROOT}/README.md" || ! -d "${PROJECT_ROOT}/containers" || ! -d "${PROJECT_ROOT}/rl" ]]; then
  echo "ERROR: run this command from the grepseek repo root, e.g.:" >&2
  echo "       bash containers/pull_images.sh" >&2
  exit 2
fi

REGISTRY="${REGISTRY:-ghcr.io/alirezasalemi7}"
TAG="${TAG:-v1-slim}"
IMAGE_DIR="${IMAGE_DIR:-${PROJECT_ROOT}/containers/images}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${PROJECT_ROOT}/containers/apptainer_cache}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${PROJECT_ROOT}/containers/apptainer_tmp}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-auto}"
INCLUDE_ALL=0
IMAGES=(grepseek)

usage() {
  cat <<'USAGE'
Usage:
  bash containers/pull_images.sh [--image IMAGE] [--all]

Defaults:
  Pulls only ghcr.io/alirezasalemi7/grepseek:v1-slim.

Options:
  --image IMAGE        Image name to pull: grepseek, grepseek-retriever, or grepseek-all.
                       May be passed more than once.
  --all                Pull grepseek, grepseek-retriever, and grepseek-all.
  --tag TAG            Image tag to pull. Default: v1-slim
  --registry REGISTRY  Image registry/namespace. Default: ghcr.io/alirezasalemi7
  --image-dir PATH     Absolute output directory for SIF files.
  -h, --help           Show this help.

Runtime:
  CONTAINER_RUNTIME=auto       Use Docker when the engine is usable, otherwise Apptainer.
  CONTAINER_RUNTIME=docker     Require a usable Docker engine.
  CONTAINER_RUNTIME=apptainer  Require Apptainer and write SIF files.

For private GHCR packages with Apptainer, set:
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

has_docker_engine() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

has_apptainer() {
  command -v apptainer >/dev/null 2>&1
}

choose_pull_runtime() {
  case "${CONTAINER_RUNTIME}" in
    docker)
      if has_docker_engine; then
        printf '%s\n' docker
      else
        echo "ERROR: CONTAINER_RUNTIME=docker requires a usable Docker engine" >&2
        exit 2
      fi
      ;;
    apptainer)
      if has_apptainer; then
        printf '%s\n' apptainer
      else
        echo "ERROR: CONTAINER_RUNTIME=apptainer requires apptainer" >&2
        exit 2
      fi
      ;;
    auto)
      if has_docker_engine; then
        printf '%s\n' docker
      elif has_apptainer; then
        printf '%s\n' apptainer
      else
        echo "ERROR: could not find a usable Docker engine or Apptainer" >&2
        exit 2
      fi
      ;;
    *)
      echo "ERROR: CONTAINER_RUNTIME must be auto, apptainer, or docker" >&2
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
    --image)
      IMAGES+=("$2")
      shift 2
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

if [[ "${INCLUDE_ALL}" == "1" ]]; then
  IMAGES=(grepseek grepseek-retriever grepseek-all)
fi

require_absolute_path PROJECT_ROOT "${PROJECT_ROOT}"
PULL_RUNTIME="$(choose_pull_runtime)"

if [[ "${PULL_RUNTIME}" == "apptainer" ]]; then
  require_absolute_path IMAGE_DIR "${IMAGE_DIR}"
  require_absolute_path APPTAINER_CACHEDIR "${APPTAINER_CACHEDIR}"
  require_absolute_path APPTAINER_TMPDIR "${APPTAINER_TMPDIR}"
  mkdir -p "${IMAGE_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

  export APPTAINER_CACHEDIR
  export APPTAINER_TMPDIR
  export TMPDIR="${APPTAINER_TMPDIR}"
fi

tag_for_file="${TAG//\//_}"
tag_for_file="${tag_for_file//:/_}"

pull_one() {
  local image_name="$1"
  local image_ref="${REGISTRY}/${image_name}:${TAG}"
  local sif_path="${IMAGE_DIR}/${image_name}_${tag_for_file}.sif"

  echo
  echo "Runtime: ${PULL_RUNTIME}"
  echo "Pulling ${image_ref}"
  if [[ "${PULL_RUNTIME}" == "docker" ]]; then
    docker pull "${image_ref}"
  else
    require_absolute_path SIF_PATH "${sif_path}"
    echo "Writing ${sif_path}"
    apptainer pull --force "${sif_path}" "docker://${image_ref}"
  fi
}

declare -A seen=()
for image_name in "${IMAGES[@]}"; do
  case "${image_name}" in
    grepseek|grepseek-retriever|grepseek-all) ;;
    *)
      echo "ERROR: unsupported image: ${image_name}" >&2
      usage >&2
      exit 2
      ;;
  esac
  if [[ -z "${seen[${image_name}]:-}" ]]; then
    pull_one "${image_name}"
    seen["${image_name}"]=1
  fi
done
