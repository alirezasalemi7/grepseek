#!/usr/bin/env bash

container_die() {
  echo "ERROR: $*" >&2
  exit 2
}

container_repo_root() {
  local root="${PWD}"
  if [[ ! -f "${root}/README.md" || ! -d "${root}/containers" || ! -d "${root}/inference" || ! -d "${root}/rl" ]]; then
    container_die "run this command from the GrepSeek repo root, e.g. bash containers/<script>.sh"
  fi
  printf '%s\n' "${root}"
}

container_abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${PWD}" "${path}"
  fi
}

container_require_absolute() {
  local name="$1"
  local path="$2"
  [[ "${path}" = /* ]] || container_die "${name} must be an absolute path, got: ${path}"
}

container_require_file() {
  local path="$1"
  [[ -f "${path}" ]] || container_die "missing file: ${path}"
}

container_require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || container_die "missing directory: ${path}"
}

container_init() {
  GREPSEEK_ROOT="$(container_repo_root)"
  REGISTRY="${REGISTRY:-ghcr.io/alirezasalemi7}"
  TAG="${TAG:-v1-slim}"
  IMAGE_DIR="${IMAGE_DIR:-${GREPSEEK_ROOT}/containers/images}"
  CACHE_DIR="${CACHE_DIR:-${GREPSEEK_ROOT}/containers/cache}"
  CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-auto}"
  CONTAINER_DISABLE_GPU="${CONTAINER_DISABLE_GPU:-0}"

  container_require_absolute GREPSEEK_ROOT "${GREPSEEK_ROOT}"
  container_require_absolute IMAGE_DIR "${IMAGE_DIR}"
  container_require_absolute CACHE_DIR "${CACHE_DIR}"

  mkdir -p \
    "${IMAGE_DIR}" \
    "${CACHE_DIR}/cuda" \
    "${CACHE_DIR}/home" \
    "${CACHE_DIR}/hf" \
    "${CACHE_DIR}/matplotlib" \
    "${CACHE_DIR}/numba" \
    "${CACHE_DIR}/outlines" \
    "${CACHE_DIR}/ray" \
    "${CACHE_DIR}/tmp" \
    "${CACHE_DIR}/triton" \
    "${CACHE_DIR}/torch" \
    "${CACHE_DIR}/vllm" \
    "${CACHE_DIR}/xdg"

  CONTAINER_EXTRA_BINDS=()
  CONTAINER_EXTRA_ENVS=()
  CONTAINER_DOCKER_PORTS=()
  CONTAINER_DOCKER_ARGS=()
  CONTAINER_APPTAINER_ARGS=()
}

container_sif_path() {
  local image_name="$1"
  local tag_for_file="${TAG//\//_}"
  tag_for_file="${tag_for_file//:/_}"
  printf '%s/%s_%s.sif\n' "${IMAGE_DIR}" "${image_name}" "${tag_for_file}"
}

container_image_ref() {
  local image_name="$1"
  printf '%s/%s:%s\n' "${REGISTRY}" "${image_name}" "${TAG}"
}

container_has_docker_engine() {
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

container_has_apptainer() {
  command -v apptainer >/dev/null 2>&1
}

container_add_bind() {
  local host_path="$1"
  local container_path="$2"
  host_path="$(container_abs_path "${host_path}")"
  container_require_absolute HOST_PATH "${host_path}"
  container_require_absolute CONTAINER_PATH "${container_path}"
  CONTAINER_EXTRA_BINDS+=("${host_path}:${container_path}")
}

container_add_existing_path_bind_same() {
  local path="$1"
  path="$(container_abs_path "${path}")"
  if [[ -d "${path}" ]]; then
    container_add_bind "${path}" "${path}"
  elif [[ -f "${path}" ]]; then
    local parent
    parent="$(dirname "${path}")"
    container_add_bind "${parent}" "${parent}"
  else
    container_die "cannot bind missing path: ${path}"
  fi
}

container_add_parent_bind_same() {
  local path="$1"
  path="$(container_abs_path "${path}")"
  local parent
  parent="$(dirname "${path}")"
  mkdir -p "${parent}"
  container_add_bind "${parent}" "${parent}"
}

container_add_env() {
  local name="$1"
  local value="$2"
  CONTAINER_EXTRA_ENVS+=("${name}=${value}")
}

container_add_port() {
  local port="$1"
  CONTAINER_DOCKER_PORTS+=("${port}:${port}")
}

container_common_envs() {
  printf '%s\n' \
    "CUDA_CACHE_PATH=/cache/cuda" \
    "XDG_CACHE_HOME=/cache/xdg" \
    "HF_HOME=/cache/hf" \
    "HF_HUB_CACHE=/cache/hf/hub" \
    "HF_HUB_DISABLE_XET=1" \
    "TRANSFORMERS_CACHE=/cache/hf/hub" \
    "SSL_CERT_FILE=/opt/envs/grepseek/lib/python3.12/site-packages/certifi/cacert.pem" \
    "REQUESTS_CA_BUNDLE=/opt/envs/grepseek/lib/python3.12/site-packages/certifi/cacert.pem" \
    "CURL_CA_BUNDLE=/opt/envs/grepseek/lib/python3.12/site-packages/certifi/cacert.pem" \
    "MPLCONFIGDIR=/cache/matplotlib" \
    "NUMBA_CACHE_DIR=/cache/numba" \
    "OUTLINES_CACHE_DIR=/cache/outlines" \
    "RAY_TMPDIR=/cache/ray" \
    "TRITON_CACHE_DIR=/cache/triton" \
    "VLLM_CACHE_ROOT=/cache/vllm" \
    "TORCHINDUCTOR_CACHE_DIR=/cache/torch" \
    "TMPDIR=/cache/tmp" \
    "TMP=/cache/tmp" \
    "TEMP=/cache/tmp" \
    "CC=/usr/bin/gcc" \
    "CXX=/usr/bin/g++"
}

container_choose_runtime() {
  case "${CONTAINER_RUNTIME}" in
    docker)
      if container_has_docker_engine; then
        printf '%s\n' docker
      else
        container_die "CONTAINER_RUNTIME=docker requires a usable Docker engine"
      fi
      ;;
    apptainer)
      if container_has_apptainer; then
        printf '%s\n' apptainer
      else
        container_die "CONTAINER_RUNTIME=apptainer requires apptainer"
      fi
      ;;
    auto)
      if container_has_docker_engine; then
        printf '%s\n' docker
      elif container_has_apptainer; then
        printf '%s\n' apptainer
      else
        container_die "could not find a usable Docker engine or Apptainer"
      fi
      ;;
    *)
      container_die "CONTAINER_RUNTIME must be auto, apptainer, or docker"
      ;;
  esac
}

container_exec() {
  local image_name="$1"
  shift

  local runtime
  runtime="$(container_choose_runtime "${image_name}")"

  local envs=()
  while IFS= read -r env_item; do
    envs+=("${env_item}")
  done < <(container_common_envs)
  envs+=("${CONTAINER_EXTRA_ENVS[@]}")

  if [[ "${runtime}" == "apptainer" ]]; then
    local sif_path
    sif_path="$(container_sif_path "${image_name}")"
    container_require_file "${sif_path}"

    local args=(exec --cleanenv)
    if [[ "${CONTAINER_DISABLE_GPU}" != "1" && "${CONTAINER_DISABLE_GPU,,}" != "true" ]]; then
      args+=(--nv)
    fi
    args+=(--bind "${GREPSEEK_ROOT}:/workspace")
    args+=(--bind "${CACHE_DIR}:/cache")
    args+=(--home "${CACHE_DIR}/home")
    local bind_spec
    for bind_spec in "${CONTAINER_EXTRA_BINDS[@]}"; do
      args+=(--bind "${bind_spec}")
    done
    local env_item
    for env_item in "${envs[@]}"; do
      args+=(--env "${env_item}")
    done
    args+=("${CONTAINER_APPTAINER_ARGS[@]}")

    echo "Runtime: apptainer"
    echo "Image:   ${sif_path}"
    exec apptainer "${args[@]}" "${sif_path}" "$@"
  fi

  local image_ref
  image_ref="$(container_image_ref "${image_name}")"
  local args=(run --rm --ipc=host)
  if [[ "${CONTAINER_DISABLE_GPU}" != "1" && "${CONTAINER_DISABLE_GPU,,}" != "true" ]]; then
    args+=(--gpus all)
  fi
  args+=(-v "${GREPSEEK_ROOT}:/workspace")
  args+=(-v "${CACHE_DIR}:/cache")
  args+=(-w /workspace)
  local bind_spec
  for bind_spec in "${CONTAINER_EXTRA_BINDS[@]}"; do
    args+=(-v "${bind_spec}")
  done
  local env_item
  for env_item in "${envs[@]}"; do
    args+=(--env "${env_item}")
  done
  local port_spec
  for port_spec in "${CONTAINER_DOCKER_PORTS[@]}"; do
    args+=(-p "${port_spec}")
  done
  args+=("${CONTAINER_DOCKER_ARGS[@]}")

  echo "Runtime: docker"
  echo "Image:   ${image_ref}"
  exec docker "${args[@]}" "${image_ref}" "$@"
}
