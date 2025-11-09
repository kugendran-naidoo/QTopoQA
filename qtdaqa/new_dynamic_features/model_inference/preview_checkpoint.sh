#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: ./preview_checkpoint.sh <config.yaml>" >&2
  exit 1
fi

CONFIG_FILE=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path
from qtdaqa.new_dynamic_features.model_inference import inference_topoqa_cpu as infer

cfg_path = Path(sys.argv[1]).resolve()
cfg = infer.load_config(cfg_path)

print("=== Inference checkpoint preview ===")
print(f"Config file   : {cfg_path}")
print(f"Training root : {cfg.training_root}")
print(f"Checkpoint    : {cfg.checkpoint_path}")
PY
