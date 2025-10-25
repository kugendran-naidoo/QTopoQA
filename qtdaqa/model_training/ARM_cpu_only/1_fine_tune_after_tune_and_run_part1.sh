#!/usr/bin/env bash
# set -euo pipefail

cfg_path="configs/sched_boost_finetune.yaml"

CKPT=$(bash ./00_locate_top_models.sh |
       head -1
      )

printf "Best checkpoint = ${CKPT}\n"

CKPT_PATH="$(pwd)/${CKPT}"

printf "Full path to checkpoint = ${CKPT_PATH}\n"

printf  "=== Fine-tuning Phase 1 ===\n"
./run_training.sh -c "${cfg_path}" -- --resume-from "${CKPT_PATH}"

