#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root regardless of invocation point.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

# Export determinism-related environment variables.  They are harmless on CPU.
export PYTHONHASHSEED=222
export PL_SEED_WORKERS=1
export TORCH_USE_DETERMINISTIC_ALGORITHMS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

ARGS=$(python - <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, 'r') as fh:
    cfg = yaml.safe_load(fh)

flag_map = {
    'graph_dir': '--graph_dir',
    'train_label_file': '--train_label_file',
    'val_label_file': '--val_label_file',
    'attention_head': '--attention_head',
    'pooling_type': '--pooling_type',
    'batch_size': '--batch_size',
    'learning_rate': '--learning_rate',
    'num_epochs': '--num_epochs',
    'accumulate_grad_batches': '--accumulate_grad_batches',
    'seed': '--seed',
    'save_dir': '--save_dir',
}

tokens = []
for key, flag in flag_map.items():
    val = cfg.get(key)
    if val is None:
        continue
    tokens.extend([flag, str(val)])

print(' '.join(tokens))
PY
"${CONFIG_FILE}")

if [[ -z "${ARGS}" ]]; then
  echo "Failed to build argument list from ${CONFIG_FILE}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

python topoqa_train/k_mac_train_topoqa.py ${ARGS}
