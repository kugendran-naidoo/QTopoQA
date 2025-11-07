#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: PYTHON='${PYTHON}' does not point to an executable interpreter." >&2
    exit 1
  fi
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: could not locate a python interpreter (python3 or python)." >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" -m qtdaqa.new_dynamic_features.model_inference.inference_topoqa_cpu "$@"
