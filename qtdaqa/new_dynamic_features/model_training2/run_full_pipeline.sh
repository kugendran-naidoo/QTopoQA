#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_MANIFEST="${SCRIPT_DIR}/manifests/run_core.yaml"
MANIFEST="${PIPELINE_MANIFEST:-${DEFAULT_MANIFEST}}"

usage() {
  cat <<'EOF'
Usage: ./run_full_pipeline.sh [--manifest PATH]

Options:
  --manifest PATH   Manifest file to use for the sweep (overrides PIPELINE_MANIFEST).
  -h, --help        Show this help message.

Environment overrides:
  PIPELINE_MANIFEST   Alternative way to point at a manifest file.
EOF
}

declare -a POSITIONAL_ARGS=()
MANIFEST_FLAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      if [[ $# -lt 2 ]]; then
        echo "[run_full_pipeline] --manifest requires a path" >&2
        usage
        exit 1
      fi
      MANIFEST_FLAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
  echo "[run_full_pipeline] Unknown arguments: ${POSITIONAL_ARGS[*]}" >&2
  usage
  exit 1
fi

if [[ -n "${MANIFEST_FLAG}" ]]; then
  MANIFEST="${MANIFEST_FLAG}"
fi

if [[ "${MANIFEST}" != /* ]]; then
  MANIFEST="${SCRIPT_DIR}/${MANIFEST}"
fi

RUN_ROOT="${SCRIPT_DIR}/training_runs2"
PHASE1_CONFIG="${PHASE1_CONFIG:-configs/sched_boost_finetune.yaml}"
PHASE2_SEEDS=(101 555 888)
PHASE2_CONFIG_BASENAME="${PHASE2_CONFIG_BASENAME:-sched_boost_finetune}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SKIP_SWEEP="${SKIP_SWEEP:-}"
SKIP_FINE="${SKIP_FINE:-}"
RESUME_FROM="${RESUME_FROM:-}"

GRAPH_DIR="${GRAPH_DIR:-}"
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-}"

if [[ -z "${GRAPH_DIR}" ]]; then
  echo "[run_full_pipeline] ERROR: GRAPH_DIR is not set. Point it at the graph builder output (…/graph_builder/output/<run>/graph_data)." >&2
  exit 1
fi
if [[ ! -d "${GRAPH_DIR}" ]]; then
  echo "[run_full_pipeline] ERROR: GRAPH_DIR does not exist: ${GRAPH_DIR}" >&2
  exit 1
fi

# Optional graph validation preflight
if [[ "${VALIDATE_GRAPHS:-0}" != "0" ]]; then
  VALIDATOR_CMD=(python -m qtdaqa.new_dynamic_features.model_training2.tools.validate_graphs --graph-dir "${GRAPH_DIR}")
  if [[ -n "${VALIDATE_GRAPHS_MANIFEST:-}" ]]; then
    VALIDATOR_CMD+=(--manifest "${VALIDATE_GRAPHS_MANIFEST}")
    if [[ "${VALIDATE_GRAPHS_CREATE_MANIFEST:-0}" != "0" && ! -f "${VALIDATE_GRAPHS_MANIFEST}" ]]; then
      VALIDATOR_CMD+=(--write-manifest)
    fi
  fi
  if [[ "${VALIDATE_GRAPHS_IGNORE_METADATA:-0}" != "0" ]]; then
    VALIDATOR_CMD+=(--ignore-metadata)
  fi
  echo "[run_full_pipeline] Validating graphs before training..."
  if ! "${VALIDATOR_CMD[@]}"; then
    echo "[run_full_pipeline] Graph validation failed; aborting." >&2
    exit 1
  fi
fi

if [[ -n "${RESUME_FROM}" && -z "${SKIP_SWEEP}" ]]; then
  echo "[run_full_pipeline][warning] RESUME_FROM is set without SKIP_SWEEP=1; the sweep will run again before fine-tuning." >&2
fi

declare -a EXTRA_OVERRIDES=()
if [[ -n "${GRAPH_DIR}" ]]; then
  EXTRA_OVERRIDES+=("--override" "paths.graph=${GRAPH_DIR}")
fi
if [[ -n "${NUM_WORKERS_OVERRIDE}" ]]; then
  EXTRA_OVERRIDES+=("--override" "dataloader.num_workers=${NUM_WORKERS_OVERRIDE}")
fi

START_TS=""
if [[ -z "${SKIP_SWEEP}" ]]; then
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "[run_full_pipeline] Manifest not found: ${MANIFEST}" >&2
    exit 1
  fi
  START_DATA="$(python - <<'PY'
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
print(now.isoformat(), int(now.timestamp()))
PY
)"
  read -r START_ISO START_TS <<< "${START_DATA}"

  echo "[run_full_pipeline] Starting sweep at ${START_ISO} using manifest ${MANIFEST}"
  if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    python -m train_cli batch --manifest "${MANIFEST}" "${EXTRA_OVERRIDES[@]}"
  else
    python -m train_cli batch --manifest "${MANIFEST}"
  fi
else
  echo "[run_full_pipeline] SKIP_SWEEP detected; skipping manifest sweep."
fi

if [[ -n "${RESUME_FROM}" ]]; then
  if ! ABS_RESUME="$(python - <<'PY' "${RESUME_FROM}"
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"; then
    echo "[run_full_pipeline] Failed to resolve RESUME_FROM path: ${RESUME_FROM}" >&2
    exit 1
  fi
  if [[ ! -f "${ABS_RESUME}" ]]; then
    echo "[run_full_pipeline] RESUME_FROM path not found: ${ABS_RESUME}" >&2
    exit 1
  fi
  BEST_CKPT="${ABS_RESUME}"
  BEST_RUN_DIR="$(dirname "$(dirname "${ABS_RESUME}")")"
  if ! RESUME_INFO_RAW="$(python - <<'PY' "${BEST_RUN_DIR}"
import sys
from pathlib import Path

from qtdaqa.new_dynamic_features.model_training2 import train_cli

run_dir = Path(sys.argv[1]).resolve()
if not run_dir.exists():
    print(f"[run_full_pipeline] ERROR: Run directory not found: {run_dir}", file=sys.stderr)
    sys.exit(1)

summary = train_cli._summarise_run(run_dir)
best_ckpt = summary.get("best_checkpoint")
if not best_ckpt:
    print(f"[run_full_pipeline] ERROR: Run {run_dir.name} produced no checkpoint.", file=sys.stderr)
    sys.exit(1)

ckpt_path = Path(best_ckpt)
if not ckpt_path.exists():
    print(f"[run_full_pipeline] ERROR: Reported checkpoint does not exist: {ckpt_path}", file=sys.stderr)
    sys.exit(1)

metric_name, metric_value = train_cli._resolve_primary_metric_value(summary)
print(str(ckpt_path))
print("" if metric_value is None else metric_value)
print(metric_name or "")
PY
  )"; then
    echo "[run_full_pipeline] Resumed run ${BEST_RUN_DIR} did not produce a usable checkpoint summary." >&2
    exit 1
  fi
  RESUME_INFO_CLEAN="${RESUME_INFO_RAW%$'\n'}"
  RESUME_CKPT="$(echo "${RESUME_INFO_CLEAN}" | sed -n '1p')"
  RESUME_SCORE="$(echo "${RESUME_INFO_CLEAN}" | sed -n '2p')"
  RESUME_METRIC="$(echo "${RESUME_INFO_CLEAN}" | sed -n '3p')"
  BEST_SCORE="${RESUME_SCORE:-n/a}"
  BEST_METRIC="${RESUME_METRIC:-val_loss}"
  if [[ -n "${RESUME_CKPT}" && "${RESUME_CKPT}" != "${BEST_CKPT}" ]]; then
    echo "[run_full_pipeline][warning] RESUME_FROM checkpoint differs from run summary: ${RESUME_CKPT}" >&2
  fi
else
  if ! BEST_INFO="$(python - <<'PY' "${START_TS}" "${RUN_ROOT}"
import json
import sys
from pathlib import Path
from datetime import datetime

from qtdaqa.new_dynamic_features.model_training2 import train_cli

run_root = Path(sys.argv[2])
start_arg = sys.argv[1]
start_dt = None
if start_arg:
    try:
        start_ts = int(start_arg)
    except ValueError:
        start_ts = None
    if start_ts and start_ts > 0:
        start_dt = datetime.fromtimestamp(start_ts)

def iter_runs(root: Path):
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir() and (p / "run_metadata.json").exists()]

candidates = []
for directory in iter_runs(run_root) + iter_runs(run_root / "history"):
    meta_path = directory / "run_metadata.json"
    if not meta_path.exists():
        continue
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    created = meta.get("created")
    if not created:
        continue
    try:
        created_dt = datetime.fromisoformat(created)
    except ValueError:
        continue
    if start_dt and created_dt < start_dt:
        continue
    summary = train_cli._summarise_run(directory)
    best_ckpt = summary.get("best_checkpoint")
    if best_ckpt is None:
        continue
    metric_name, metric_value = train_cli._resolve_primary_metric_value(summary)
    if metric_value is None:
        continue
    candidates.append((float(metric_value), metric_name, best_ckpt, str(directory)))

if not candidates:
    print("ERROR: No new training runs detected after sweep.", file=sys.stderr)
    sys.exit(1)

best_score, metric_name, best_ckpt, best_run = sorted(candidates)[0]
print(best_ckpt)
print(best_run)
print(best_score)
print(metric_name)
PY
  )"; then
    echo "[run_full_pipeline] Unable to identify best checkpoint from sweep results." >&2
    exit 1
  fi

  if [[ -z "${BEST_INFO}" ]]; then
    echo "[run_full_pipeline] Unable to identify best checkpoint." >&2
    exit 1
  fi

  BEST_CKPT="$(echo "${BEST_INFO}" | sed -n '1p')"
  BEST_RUN_DIR="$(echo "${BEST_INFO}" | sed -n '2p')"
  BEST_SCORE="$(echo "${BEST_INFO}" | sed -n '3p')"
  BEST_METRIC="$(echo "${BEST_INFO}" | sed -n '4p')"
fi

echo "[run_full_pipeline] Best run: ${BEST_RUN_DIR} (${BEST_METRIC:-val_loss}=${BEST_SCORE:-n/a})"
echo "[run_full_pipeline] Best checkpoint: ${BEST_CKPT}"

if [[ ! -f "${BEST_CKPT}" ]]; then
  echo "[run_full_pipeline] Best checkpoint path not found: ${BEST_CKPT}" >&2
  exit 1
fi

if [[ -n "${SKIP_FINE}" ]]; then
  echo "[run_full_pipeline] SKIP_FINE detected; stopping after coarse sweep."
  exit 0
fi

BASE_RUN_NAME="$(basename "${BEST_RUN_DIR}")"
PHASE1_RUN_NAME="${BASE_RUN_NAME}_phase1"

echo "[run_full_pipeline] Phase 1 fine-tuning -> ${PHASE1_RUN_NAME}"
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  python -m train_cli run \
    --config "${PHASE1_CONFIG}" \
    --run-name "${PHASE1_RUN_NAME}" \
    --resume-from "${BEST_CKPT}" \
    "${EXTRA_OVERRIDES[@]}"
else
  python -m train_cli run \
    --config "${PHASE1_CONFIG}" \
    --run-name "${PHASE1_RUN_NAME}" \
    --resume-from "${BEST_CKPT}"
fi

PHASE1_RUN_DIR="${RUN_ROOT}/${PHASE1_RUN_NAME}"
if [[ ! -d "${PHASE1_RUN_DIR}" ]]; then
  echo "[run_full_pipeline] Phase 1 run directory not found: ${PHASE1_RUN_DIR}" >&2
  exit 1
fi

if ! PHASE1_INFO_RAW="$(python - <<'PY' "${PHASE1_RUN_DIR}"
import sys
from pathlib import Path

from qtdaqa.new_dynamic_features.model_training2 import train_cli

run_dir = Path(sys.argv[1]).resolve()
if not run_dir.exists():
    print(f"[run_full_pipeline] ERROR: Run directory not found: {run_dir}", file=sys.stderr)
    sys.exit(1)

summary = train_cli._summarise_run(run_dir)
best_ckpt = summary.get("best_checkpoint")
if not best_ckpt:
    print(f"[run_full_pipeline] ERROR: Run {run_dir.name} produced no checkpoint.", file=sys.stderr)
    sys.exit(1)

ckpt_path = Path(best_ckpt)
if not ckpt_path.exists():
    print(f"[run_full_pipeline] ERROR: Reported checkpoint does not exist: {ckpt_path}", file=sys.stderr)
    sys.exit(1)

metric_name, metric_value = train_cli._resolve_primary_metric_value(summary)
print(str(ckpt_path))
print("" if metric_value is None else metric_value)
print(metric_name or "")
PY
)"; then
  echo "[run_full_pipeline] Phase 1 fine-tune did not produce a usable checkpoint." >&2
  exit 1
fi

PHASE1_INFO_CLEAN="${PHASE1_INFO_RAW%$'\n'}"
PHASE1_BEST_CKPT="$(echo "${PHASE1_INFO_CLEAN}" | sed -n '1p')"
PHASE1_BEST_SCORE="$(echo "${PHASE1_INFO_CLEAN}" | sed -n '2p')"
PHASE1_BEST_METRIC="$(echo "${PHASE1_INFO_CLEAN}" | sed -n '3p')"
PHASE1_BEST_CKPT="${PHASE1_BEST_CKPT:-}"
PHASE1_BEST_SCORE="${PHASE1_BEST_SCORE:-n/a}"
PHASE1_BEST_METRIC="${PHASE1_BEST_METRIC:-val_loss}"
if [[ -z "${PHASE1_BEST_CKPT}" ]]; then
  echo "[run_full_pipeline] Phase 1 summary did not return a checkpoint path." >&2
  exit 1
fi
echo "[run_full_pipeline] Phase 1 best checkpoint: ${PHASE1_BEST_CKPT} (${PHASE1_BEST_METRIC}=${PHASE1_BEST_SCORE})"

for seed in "${PHASE2_SEEDS[@]}"; do
  PHASE2_CONFIG="${SCRIPT_DIR}/configs/${PHASE2_CONFIG_BASENAME}_seed${seed}.yaml"
  if [[ ! -f "${PHASE2_CONFIG}" ]]; then
    echo "[run_full_pipeline] Skipping seed ${seed} (config not found: ${PHASE2_CONFIG})" >&2
    continue
  fi
  RUN_NAME="${BASE_RUN_NAME}_phase2_seed${seed}"
  echo "[run_full_pipeline] Phase 2 fine-tuning (seed ${seed}) -> ${RUN_NAME}"
  if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
    python -m train_cli run \
      --config "${PHASE2_CONFIG}" \
      --run-name "${RUN_NAME}" \
      --resume-from "${PHASE1_BEST_CKPT}" \
      "${EXTRA_OVERRIDES[@]}"
  else
    python -m train_cli run \
      --config "${PHASE2_CONFIG}" \
      --run-name "${RUN_NAME}" \
      --resume-from "${PHASE1_BEST_CKPT}"
  fi

  PHASE2_RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
  if [[ ! -d "${PHASE2_RUN_DIR}" ]]; then
    echo "[run_full_pipeline] Phase 2 run directory not found: ${PHASE2_RUN_DIR}" >&2
    exit 1
  fi

  if ! PHASE2_INFO_RAW="$(python - <<'PY' "${PHASE2_RUN_DIR}"
import sys
from pathlib import Path

from qtdaqa.new_dynamic_features.model_training2 import train_cli

run_dir = Path(sys.argv[1]).resolve()
if not run_dir.exists():
    print(f"[run_full_pipeline] ERROR: Run directory not found: {run_dir}", file=sys.stderr)
    sys.exit(1)

summary = train_cli._summarise_run(run_dir)
best_ckpt = summary.get("best_checkpoint")
if not best_ckpt:
    print(f"[run_full_pipeline] ERROR: Run {run_dir.name} produced no checkpoint.", file=sys.stderr)
    sys.exit(1)

ckpt_path = Path(best_ckpt)
if not ckpt_path.exists():
    print(f"[run_full_pipeline] ERROR: Reported checkpoint does not exist: {ckpt_path}", file=sys.stderr)
    sys.exit(1)

metric_name, metric_value = train_cli._resolve_primary_metric_value(summary)
print(str(ckpt_path))
print("" if metric_value is None else metric_value)
print(metric_name or "")
PY
  )"; then
    echo "[run_full_pipeline] Phase 2 fine-tune (seed ${seed}) did not produce a usable checkpoint." >&2
    exit 1
  fi

  PHASE2_INFO_CLEAN="${PHASE2_INFO_RAW%$'\n'}"
  PHASE2_BEST_CKPT="$(echo "${PHASE2_INFO_CLEAN}" | sed -n '1p')"
  PHASE2_BEST_SCORE="$(echo "${PHASE2_INFO_CLEAN}" | sed -n '2p')"
  PHASE2_BEST_METRIC="$(echo "${PHASE2_INFO_CLEAN}" | sed -n '3p')"
  PHASE2_BEST_CKPT="${PHASE2_BEST_CKPT:-}"
  PHASE2_BEST_SCORE="${PHASE2_BEST_SCORE:-n/a}"
  PHASE2_BEST_METRIC="${PHASE2_BEST_METRIC:-val_loss}"
  if [[ -z "${PHASE2_BEST_CKPT}" ]]; then
    echo "[run_full_pipeline] Phase 2 summary for seed ${seed} did not return a checkpoint path." >&2
    exit 1
  fi
  echo "[run_full_pipeline] Phase 2 (seed ${seed}) best checkpoint: ${PHASE2_BEST_CKPT} (${PHASE2_BEST_METRIC}=${PHASE2_BEST_SCORE})"
done

ISO_TS="$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date '+%Y-%m-%dT%H:%M:%SZ')"
EMA_BACKFILL="${EMA_BACKFILL:-1}"
EMA_BACKFILL_FORCE="${EMA_BACKFILL_FORCE:-0}"
if [[ "${EMA_BACKFILL}" != "0" ]]; then
  echo "[run_full_pipeline] Backfilling EMA metrics (if missing)..."
  EMA_CMD=(python -m qtdaqa.new_dynamic_features.model_training2.tools.ema_backfill_eval --training-root "${RUN_ROOT}")
  if [[ "${EMA_BACKFILL_FORCE}" != "0" ]]; then
    EMA_CMD+=(--force)
  fi
  if ! "${EMA_CMD[@]}"; then
    echo "[run_full_pipeline][warning] EMA backfill failed; continuing." >&2
  fi
else
  echo "[run_full_pipeline] EMA backfill skipped (EMA_BACKFILL=0)."
fi
echo "[run_full_pipeline] Pipeline complete at ${ISO_TS}"
