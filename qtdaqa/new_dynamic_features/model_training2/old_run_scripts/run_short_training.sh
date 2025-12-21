#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="bundleb_checkpoint_demo"
RUN_DIR="training_runs2/${RUN_NAME}"

if [[ -d "${RUN_DIR}" ]]; then
  rm -rf "${RUN_DIR}"
fi

python -m train_cli run \
  --config configs/sched_boost_seed777.yaml \
  --run-name "${RUN_NAME}" \
  --override trainer.num_epochs=5 \
  --limit-train-batches 0.1 \
  --limit-val-batches 0.1
