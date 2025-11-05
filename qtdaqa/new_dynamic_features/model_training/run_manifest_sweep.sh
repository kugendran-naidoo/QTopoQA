#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_BUNDLE="core"
MANIFEST_PATH=""
BUNDLE_NAME="${DEFAULT_BUNDLE}"
FAST_DEV_FLAG=0

usage() {
  cat <<'EOF'
Usage: ./run_manifest_sweep.sh [--bundle core|extended|experiments] [--manifest PATH]
                              [--fast-dev] [--help] [-- <additional train_cli args>]

Runs `python -m train_cli batch` with the requested manifest.  If both --bundle and
--manifest are supplied, the explicit --manifest path wins.

Options:
  --bundle NAME   Shortcut for manifests/run_<name>.yaml (core|extended|experiments).
                  Defaults to "core".
  --manifest PATH Use a custom manifest file instead of the bundled presets.
  --fast-dev      Injects `--override fast_dev_run=true` so every job runs in
                  Lightning's one-batch smoke-test mode.
  --help          Show this message.

Any arguments after `--` are forwarded to `python -m train_cli batch` unchanged.
EOF
}

EXTRA_ARGS=()
declare -a EXTRA_ARGS
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      if [[ $# -lt 2 ]]; then
        echo "[run_manifest_sweep] --bundle requires a value (core|extended|experiments)." >&2
        usage
        exit 1
      fi
      BUNDLE_NAME="$2"
      shift 2
      ;;
    --manifest)
      if [[ $# -lt 2 ]]; then
        echo "[run_manifest_sweep] --manifest requires a path." >&2
        usage
        exit 1
      fi
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --fast-dev)
      FAST_DEV_FLAG=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${MANIFEST_PATH}" ]]; then
  case "${BUNDLE_NAME}" in
    core)
      MANIFEST_PATH="${SCRIPT_DIR}/manifests/run_core.yaml"
      ;;
    extended)
      MANIFEST_PATH="${SCRIPT_DIR}/manifests/run_extended.yaml"
      ;;
    experiments|experiment|smoke)
      MANIFEST_PATH="${SCRIPT_DIR}/manifests/run_experiments.yaml"
      ;;
    *)
      echo "[run_manifest_sweep] Unknown bundle '${BUNDLE_NAME}'. Use core, extended, or experiments." >&2
      usage
      exit 1
      ;;
  esac
fi

if [[ "${MANIFEST_PATH}" != /* ]]; then
  MANIFEST_PATH="${SCRIPT_DIR}/${MANIFEST_PATH}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[run_manifest_sweep] Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${REPO_ROOT}/venv_qtopo/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PARENT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
  ALT_BIN="${PARENT_ROOT}/venv_qtopo/bin/python"
  if [[ -x "${ALT_BIN}" ]]; then
    PYTHON_BIN="${ALT_BIN}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
echo "[run_manifest_sweep] Using manifest ${MANIFEST_PATH}"
CMD=("${PYTHON_BIN}" -m train_cli batch --manifest "${MANIFEST_PATH}")
if [[ ${FAST_DEV_FLAG} -eq 1 ]]; then
  CMD+=(--override fast_dev_run=true)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi
"${CMD[@]}"
