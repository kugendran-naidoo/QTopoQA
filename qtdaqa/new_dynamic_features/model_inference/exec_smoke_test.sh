#!/usr/bin/env bash
set -euo pipefail

# Repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths for the smoke roundtrip
CKPT="$REPO_ROOT/qtdaqa/new_dynamic_features/model_training/training_runs/smoke_sched_boost_seed222_2025-12-07_21-51-17/model_checkpoints/best.ckpt"
CONFIG="$REPO_ROOT/qtdaqa/new_dynamic_features/model_inference/config.yaml.BM55-AF2.smoke_test"
GRAPH_DIR="$REPO_ROOT/qtdaqa/new_dynamic_features/graph_builder2/output/smoke_test/graph_data"
WORK_BASE="/tmp/qtopo_smoke_infer/work"
RESULTS="/tmp/qtopo_smoke_infer/results"
DATASET="smoke"

# Ensure work dir and reuse of existing smoke graphs
mkdir -p "$WORK_BASE/$DATASET"
mkdir -p "$RESULTS"
if [[ ! -e "$WORK_BASE/$DATASET/graph_data" ]]; then
  ln -s "$GRAPH_DIR" "$WORK_BASE/$DATASET/graph_data"
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Run inference with schema check, reusing the existing smoke graphs and small labels.
bash "$REPO_ROOT/qtdaqa/new_dynamic_features/model_inference/run_model_inference.sh" \
  --dataset-name "$DATASET" \
  --config "$CONFIG" \
  --checkpoint-path "$CKPT" \
  --work-dir "$WORK_BASE" \
  --results-dir "$RESULTS" \
  --label-file "$REPO_ROOT/qtdaqa/new_dynamic_features/model_training/smoke_test.validate.csv" \
  --reuse-existing-graphs \
  --check-schema

echo "Smoke inference/schema check complete. Work dir: $WORK_BASE, Results: $RESULTS"
