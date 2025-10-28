#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="config.yaml"

python "${SCRIPT_DIR}/inference_topoqa_cpu.py" --config "${SCRIPT_DIR}/${CONFIG}"
