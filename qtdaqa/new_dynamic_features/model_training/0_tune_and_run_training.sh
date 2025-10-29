#!/usr/bin/env bash
set -euo pipefail

CFG_DIR="configs"
mkdir -p "$CFG_DIR"

trials=(
  sched_boost              # Rank 1: lr=0.005, patience 6/45, factor 0.2, epochs 260
  sched_boost_lr035        # Rank 2: same as above with lr=0.0035
  sched_boost_finetune     # Rank 3: fine-tune stage (requires resume-from)
  large_batch_sched_boost  # Rank 4: increase batch size
)

# seeds=(222 777 1337)
# originally tried 3, but 222 worked best on original features
# originally tried 3, but 777 worked best on 24d edge features
# seeds=(777)
seeds=(222 777 1337)

patch_yaml() {
  python - "$1" "$2" <<'PY'
import sys, yaml
cfg_path, payload_yaml = sys.argv[1:]
payload = yaml.safe_load(payload_yaml)
with open(cfg_path) as fh:
    data = yaml.safe_load(fh) or {}
data.update(payload)
with open(cfg_path, "w") as fh:
    yaml.safe_dump(data, fh)
PY
}

for trial in "${trials[@]}"; do
  case "$trial" in
    sched_boost)
      lr=0.005
      lr_pat=6
      lr_factor=0.2
      early_pat=45
      accum=16
      batch=16
      epochs=260
      seed_override=""
      ;;
    sched_boost_lr035)
      lr=0.0035
      lr_pat=6
      lr_factor=0.2
      early_pat=45
      accum=16
      batch=16
      epochs=260
      seed_override=""
      ;;
    sched_boost_finetune)
      lr=0.001
      lr_pat=4
      lr_factor=0.2
      early_pat=20
      accum=16
      batch=16
      epochs=320
      seed_override=222
      ;;
    large_batch_sched_boost)
      lr=0.005
      lr_pat=6
      lr_factor=0.2
      early_pat=45
      accum=16
      batch=24
      epochs=260
      seed_override=""
      ;;
    *)
      echo "Unknown trial: $trial" >&2
      exit 1
      ;;
  esac

  cfg="${CFG_DIR}/${trial}.yaml"
  cp config.yaml "$cfg"

  patch_yaml "$cfg" "$(cat <<EOF
learning_rate: ${lr}
lr_scheduler_patience: ${lr_pat}
lr_scheduler_factor: ${lr_factor}
early_stopping_patience: ${early_pat}
accumulate_grad_batches: ${accum}
batch_size: ${batch}
num_epochs: ${epochs}
EOF
)"

  if [[ "${seed_override:-}" != "" ]]; then
    patch_yaml "$cfg" "$(cat <<EOF
seed: ${seed_override}
EOF
)"
  fi

  if [[ "$trial" == "sched_boost_finetune" ]]; then
    echo "=== ${trial}: configure manually (resume-from required) ==="
    echo "Config written to ${cfg}. Run fine-tune via:"
    echo "  ./run_training.sh --trial \"${trial}\" -c \"${cfg}\" -- --resume-from <best-checkpoint>"
    continue
  fi

  for seed in "${seeds[@]}"; do
    cfg_seed="${CFG_DIR}/${trial}_seed${seed}.yaml"
    cp "$cfg" "$cfg_seed"
    patch_yaml "$cfg_seed" "$(cat <<EOF
seed: ${seed}
EOF
)"
    trial_label="${trial}_seed${seed}"
    echo "=== Running ${trial} with seed ${seed} ==="
    ./run_training.sh --trial "${trial_label}" -c "$cfg_seed"
  done
done
