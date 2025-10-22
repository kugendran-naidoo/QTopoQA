#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="configs/sched_boost_finetune.yaml"
SEEDS=(101 555 888)
CKPT_PATH="$(pwd)/training_runs/history/training_run_2025-10-21_23-52-10/model_checkpoints/model.0.05257.chkpt"

for seed in "${SEEDS[@]}"; do
  cfg_path="configs/sched_boost_finetune_seed${seed}.yaml"
  cp "${BASE_CFG}" "${cfg_path}"
  python - <<'PY' "${cfg_path}" "${seed}"
import sys, yaml
cfg_path, seed = sys.argv[1], int(sys.argv[2])
with open(cfg_path) as fh:
    cfg = yaml.safe_load(fh)
cfg["seed"] = seed
with open(cfg_path, "w") as fh:
    yaml.safe_dump(cfg, fh)
PY

  echo "=== Fine-tuning with seed ${seed} ==="
  ./run_training.sh -c "${cfg_path}" -- --resume-from "${CKPT_PATH}"
done
