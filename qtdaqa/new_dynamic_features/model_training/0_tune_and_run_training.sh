#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/manifests/tune_and_run.yaml"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Expected manifest not found: ${MANIFEST}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
exec "${PYTHON_BIN}" -m train_cli batch --manifest "${MANIFEST}" "$@"
