#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_training.sh -c CONFIG [--trial LABEL] [--run-name NAME] [-- extra_args]

This wrapper now delegates to the Python CLI:
  python -m train_cli run --config CONFIG [...options]

Arguments after "--" are forwarded to the underlying trainer. Known flags
such as --resume-from or --fast-dev-run are mapped automatically.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

CONFIG_FILE=""
TRIAL_LABEL=""
RUN_NAME=""
FORWARDED=()

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
      FORWARDED=("$@")
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

RESUME_FROM=""
LIMIT_TRAIN=""
LIMIT_VAL=""
FAST_DEV_RUN="0"
LOG_LR="0"
TRAINER_ARGS=()

while [[ ${#FORWARDED[@]} -gt 0 ]]; do
  token="${FORWARDED[0]}"
  case "${token}" in
    --resume-from)
      if [[ ${#FORWARDED[@]} -lt 2 ]]; then
        echo "Missing value for --resume-from" >&2
        exit 1
      fi
      RESUME_FROM="${FORWARDED[1]:-}"
      FORWARDED=(${FORWARDED[@]:2})
      ;;
    --limit-train-batches)
      if [[ ${#FORWARDED[@]} -lt 2 ]]; then
        echo "Missing value for --limit-train-batches" >&2
        exit 1
      fi
      LIMIT_TRAIN="${FORWARDED[1]:-}"
      FORWARDED=(${FORWARDED[@]:2})
      ;;
    --limit-val-batches)
      if [[ ${#FORWARDED[@]} -lt 2 ]]; then
        echo "Missing value for --limit-val-batches" >&2
        exit 1
      fi
      LIMIT_VAL="${FORWARDED[1]:-}"
      FORWARDED=(${FORWARDED[@]:2})
      ;;
    --fast-dev-run)
      FAST_DEV_RUN="1"
      FORWARDED=(${FORWARDED[@]:1})
      ;;
    --log-lr)
      LOG_LR="1"
      FORWARDED=(${FORWARDED[@]:1})
      ;;
    "")
      FORWARDED=(${FORWARDED[@]:1})
      ;;
    *)
      TRAINER_ARGS+=("${token}")
      FORWARDED=(${FORWARDED[@]:1})
      ;;
  esac
done

PYTHON_BIN="${PYTHON:-python}"
CLI_ARGS=("${PYTHON_BIN}" -m train_cli run --config "${CONFIG_PATH}")

if [[ -n "${RUN_NAME}" ]]; then
  CLI_ARGS+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${TRIAL_LABEL}" ]]; then
  CLI_ARGS+=(--trial-label "${TRIAL_LABEL}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  CLI_ARGS+=(--resume-from "${RESUME_FROM}")
fi
if [[ -n "${LIMIT_TRAIN}" ]]; then
  CLI_ARGS+=(--limit-train-batches "${LIMIT_TRAIN}")
fi
if [[ -n "${LIMIT_VAL}" ]]; then
  CLI_ARGS+=(--limit-val-batches "${LIMIT_VAL}")
fi
if [[ "${FAST_DEV_RUN}" == "1" ]]; then
  CLI_ARGS+=(--fast-dev-run)
fi
if [[ "${LOG_LR}" == "1" ]]; then
  CLI_ARGS+=(--log-lr)
fi

for arg in "${TRAINER_ARGS[@]}"; do
  CLI_ARGS+=(--trainer-arg "${arg}")
done

exec "${CLI_ARGS[@]}"
