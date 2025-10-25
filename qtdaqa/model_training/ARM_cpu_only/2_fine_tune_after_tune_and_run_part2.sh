#!/usr/bin/env bash
# set -euo pipefail

BASE_CFG="configs/sched_boost_finetune.yaml"

SEEDS=(101 555 888)

CKPT=$(bash ./00_locate_top_models.sh |
       head -1
      )

printf "Best checkpoint = ${CKPT}\n"


CKPT_PATH="$(pwd)/${CKPT}"

printf "Full path to checkpoint = ${CKPT_PATH}\n"

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

  printf "=== Fine-tuning Phase 2 - seed ${seed} ===\n"
  ./run_training.sh -c "${cfg_path}" -- --resume-from "${CKPT_PATH}"

done
