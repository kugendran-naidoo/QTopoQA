#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_training.sh -c CONFIG [--trial LABEL] [--run-name NAME] [-- python_args...]

Mandatory options:
  -c, --config PATH     Path to the YAML configuration describing training inputs,
                        hyper-parameters, and runtime settings.

Optional passthrough:
  Any arguments after the configuration flag are forwarded to train_topoqa_cpu.py.
  For example:
    ./run_training.sh -c config.yaml --fast-dev-run
    ./run_training.sh -c config.yaml -- --limit-train-batches 0.25 --limit-val-batches 0.5

Wrapper options:
  --trial LABEL         Identifier stored in training.log for bookkeeping.
  --run-name NAME       Custom directory name under training_runs/ (default timestamp).

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
TRIAL_LABEL=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --trial)
      TRIAL_LABEL="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
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
      if [[ -n "${CONFIG_FILE}" ]]; then
        PASSTHRU=("$@")
        break
      else
        echo "Unknown option: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Error: configuration file must be specified with -c/--config." >&2
  usage
  exit 1
fi

if [[ -n "${RUN_NAME}" ]]; then
  if [[ "${RUN_NAME}" == */* ]]; then
    echo "Error: --run-name must not include '/' characters." >&2
    exit 1
  fi
  if [[ "${RUN_NAME}" =~ [[:space:]] ]]; then
    echo "Error: --run-name must not include whitespace." >&2
    exit 1
  fi
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

CONFIG_SUMMARY_RAW="$(
python - "$CONFIG_PATH" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle) or {}
def _fmt(key):
    value = data.get(key)
    if value is None:
        return "N/A"
    return value
print(f"{_fmt('learning_rate')}|{_fmt('num_epochs')}|{_fmt('seed')}")
PY
)"
IFS='|' read -r CFG_LR CFG_EPOCHS CFG_SEED <<<"${CONFIG_SUMMARY_RAW}"
echo "Config summary: lr=${CFG_LR}, epochs=${CFG_EPOCHS}, seed=${CFG_SEED}"

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

run_label="${RUN_NAME:-training_run_${timestamp}}"
RUN_DIR="${RUN_ROOT}/${run_label}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Error: run directory already exists: ${RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/config"
ln -sfn "${RUN_DIR}" "${RUN_ROOT}/latest"

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

GIT_COMMIT="unknown"
GIT_DIRTY="false"
if command -v git >/dev/null 2>&1; then
  if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
      GIT_DIRTY="true"
    fi
  fi
fi

cd "${REPO_ROOT}"

PYTHON_CMD=(
  python
  "${SCRIPT_DIR}/model_train_topoqa_cpu.py"
  --config
  "${RUN_DIR}/config/config.yaml"
  --run-dir
  "${RUN_DIR}"
  --git-commit
  "${GIT_COMMIT}"
)

if [[ -n "${TRIAL_LABEL}" ]]; then
  PYTHON_CMD+=(--trial-label "${TRIAL_LABEL}")
fi
if [[ "${GIT_DIRTY}" == "true" ]]; then
  PYTHON_CMD+=(--git-dirty)
fi
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  PYTHON_CMD+=("${PASSTHRU[@]}")
fi

CONSOLE_LOG="${RUN_DIR}/training_console.log"
echo "Console output will be streamed to ${CONSOLE_LOG}"
"${PYTHON_CMD[@]}" 2>&1 | tee "${CONSOLE_LOG}"
