#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GRAPH_DIR="${GRAPH_DIR:-}"
if [[ -z "${GRAPH_DIR}" ]]; then
  echo "[smoke_test] ERROR: GRAPH_DIR is not set. Point it at graph_builder2 output (…/graph_data)." >&2
  exit 1
fi
if [[ ! -d "${GRAPH_DIR}" ]]; then
  echo "[smoke_test] ERROR: GRAPH_DIR does not exist: ${GRAPH_DIR}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m qtdaqa.new_dynamic_features.model_training2.train_cli run \
  --config configs/sched_boost_seed222.yaml \
  --run-name smoke_test \
  --fast-dev-run \
  --override "paths.graph=${GRAPH_DIR}" \
  --override "selection.primary_metric=val_loss"
