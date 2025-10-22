#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_training.sh -c CONFIG [-- python_args...]

Mandatory options:
  -c, --config PATH     Path to the YAML configuration describing training inputs,
                        hyper-parameters, and runtime settings.

Optional passthrough:
  Any arguments after the configuration flag are forwarded to train_topoqa_cpu.py.
  For example:
    ./run_training.sh -c config.yaml -- --fast-dev-run
    ./run_training.sh -c config.yaml -- --limit-train-batches 0.25 --limit-val-batches 0.5

Forwarded Python options of note:
  --fast-dev-run               Lightning dry run (1 batch train/val).
  --limit-train-batches VALUE  Limit training batches (float fraction or int count).
  --limit-val-batches VALUE    Limit validation batches (float fraction or int count).

Environment defaults (override via shell exports if required):
  PYTHONHASHSEED=222
  PL_SEED_WORKERS=1
  TORCH_USE_DETERMINISTIC_ALGORITHMS=1
  CUBLAS_WORKSPACE_CONFIG=:16:8
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

CONFIG_FILE=""
PASSTHRU=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --)
      shift
      PASSTHRU=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Error: configuration file must be specified with -c/--config." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${CONFIG_FILE}" != /* ]]; then
  CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_FILE}"
else
  CONFIG_PATH="${CONFIG_FILE}"
fi
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Configuration file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_ROOT="${SCRIPT_DIR}/training_runs"
mkdir -p "${RUN_ROOT}"

history_root="${RUN_ROOT}/history"
mkdir -p "${history_root}"
shopt -s nullglob
for prev_run in "${RUN_ROOT}"/training_run_*; do
  if [[ -d "${prev_run}" ]]; then
    mv "${prev_run}" "${history_root}/$(basename "${prev_run}")"
  fi
done
shopt -u nullglob

RUN_DIR="${RUN_ROOT}/training_run_${timestamp}"
mkdir -p "${RUN_DIR}/config"

cp "${CONFIG_PATH}" "${RUN_DIR}/config/config.yaml"
REQUIREMENTS_SRC="${SCRIPT_DIR}/requirements.txt"
if [[ -f "${REQUIREMENTS_SRC}" ]]; then
  cp "${REQUIREMENTS_SRC}" "${RUN_DIR}/config/requirements.txt"
else
  echo "Warning: requirements.txt not found at ${REQUIREMENTS_SRC}; skipping copy." >&2
fi

export PYTHONHASHSEED=${PYTHONHASHSEED:-222}
export PL_SEED_WORKERS=${PL_SEED_WORKERS:-1}
export TORCH_USE_DETERMINISTIC_ALGORITHMS=${TORCH_USE_DETERMINISTIC_ALGORITHMS:-1}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

cd "${REPO_ROOT}"

if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  python "${SCRIPT_DIR}/model_train_topoqa_cpu.py" --config "${RUN_DIR}/config/config.yaml" --run-dir "${RUN_DIR}" "${PASSTHRU[@]}"
else
  python "${SCRIPT_DIR}/model_train_topoqa_cpu.py" --config "${RUN_DIR}/config/config.yaml" --run-dir "${RUN_DIR}"
fi
