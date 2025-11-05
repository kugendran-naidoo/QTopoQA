#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

usage() {
  cat <<'EOF'
Usage: ramdisk_run_pipeline.sh [options] [-- <pipeline args>]

Options:
  --config PATH              Path to RAM-disk settings YAML (default: configs/ramdisk.yaml)
  --graph-dir PATH           Override graph source directory copied onto the RAM disk
  --manifest PATH            Manifest forwarded to run_full_pipeline.sh
  --skip-baseline            Skip the baseline (non-RAM) timing run
  --baseline-num-workers N   Override num_workers for the baseline run
  --ram-num-workers N        Override num_workers for the RAM-disk run
  --size SIZE                Override RAM disk size (default from config)
  --headroom SIZE            Override RAM disk headroom (default from config)
  --name NAME                Override RAM disk volume name (default from config)
  -h, --help                 Show this help message

Any arguments after '--' are forwarded to ./run_full_pipeline.sh for both runs.

Environment overrides:
  PIPELINE_MANIFEST          Alternate way to select the manifest.
EOF
}

CONFIG="configs/ramdisk.yaml"
GRAPH_OVERRIDE=""
MANIFEST_OVERRIDE=""
SKIP_BASELINE=0
BASELINE_NUM_WORKERS_OVERRIDE=""
RAM_NUM_WORKERS_OVERRIDE=""
RAM_SIZE_OVERRIDE=""
RAM_HEADROOM_OVERRIDE=""
RAM_NAME_OVERRIDE=""

PIPELINE_MANIFEST_WAS_SET=0
ORIGINAL_PIPELINE_MANIFEST=""
if [[ "${PIPELINE_MANIFEST-__UNSET__}" != "__UNSET__" ]]; then
  PIPELINE_MANIFEST_WAS_SET=1
  ORIGINAL_PIPELINE_MANIFEST="${PIPELINE_MANIFEST}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --graph-dir)
      GRAPH_OVERRIDE="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST_OVERRIDE="$2"
      shift 2
      ;;
    --skip-baseline)
      SKIP_BASELINE=1
      shift
      ;;
    --baseline-num-workers)
      BASELINE_NUM_WORKERS_OVERRIDE="$2"
      shift 2
      ;;
    --ram-num-workers)
      RAM_NUM_WORKERS_OVERRIDE="$2"
      shift 2
      ;;
    --size)
      RAM_SIZE_OVERRIDE="$2"
      shift 2
      ;;
    --headroom)
      RAM_HEADROOM_OVERRIDE="$2"
      shift 2
      ;;
    --name)
      RAM_NAME_OVERRIDE="$2"
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
      echo "[ramdisk] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PIPELINE_ARGS=("$@")
PIPELINE_HAS_ARGS=$#

GRAPH_DIR_WAS_SET=0
ORIGINAL_GRAPH_DIR=""
GRAPH_SOURCE_FROM_ENV=0
if [[ "${GRAPH_DIR-__UNSET__}" != "__UNSET__" ]]; then
  GRAPH_DIR_WAS_SET=1
  ORIGINAL_GRAPH_DIR="${GRAPH_DIR}"
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ramdisk] Config file not found: ${CONFIG}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ramdisk] rsync is required but not available on PATH." >&2
  exit 1
fi

if [[ ! -x "${SCRIPT_DIR}/mac_mk_safe_ramdisk.sh" ]]; then
  echo "[ramdisk] mac_mk_safe_ramdisk.sh is not executable or missing." >&2
  exit 1
fi

eval "$(
  python - "$CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).resolve()
cfg = {}
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

ram_cfg = cfg.get("ramdisk", {}) or {}
graph_source = cfg.get("graph_source")
if graph_source:
    graph_source = str((config_path.parent / Path(graph_source)).resolve())

num_workers_cfg = cfg.get("num_workers", {}) or {}

def emit(name: str, value):
    if value is None:
        return
    print(f'{name}="{value}"')

emit("CFG_RAM_SIZE", ram_cfg.get("size", "12G"))
emit("CFG_RAM_HEADROOM", ram_cfg.get("headroom", "4G"))
emit("CFG_RAM_NAME", ram_cfg.get("name", "QTopoRAM"))
emit("CFG_GRAPH_SOURCE", graph_source)
emit("CFG_BASELINE_NUM_WORKERS", num_workers_cfg.get("baseline"))
emit("CFG_RAM_NUM_WORKERS", num_workers_cfg.get("ramdisk"))
PY
)"

RAM_SIZE="${CFG_RAM_SIZE:-12G}"
RAM_HEADROOM="${CFG_RAM_HEADROOM:-4G}"
RAM_NAME="${CFG_RAM_NAME:-QTopoRAM}"
if [[ -n "${RAM_SIZE_OVERRIDE}" ]]; then
  RAM_SIZE="${RAM_SIZE_OVERRIDE}"
fi
if [[ -n "${RAM_HEADROOM_OVERRIDE}" ]]; then
  RAM_HEADROOM="${RAM_HEADROOM_OVERRIDE}"
fi
if [[ -n "${RAM_NAME_OVERRIDE}" ]]; then
  RAM_NAME="${RAM_NAME_OVERRIDE}"
fi

GRAPH_SOURCE="${GRAPH_OVERRIDE:-${CFG_GRAPH_SOURCE:-}}"
if [[ -z "${GRAPH_SOURCE}" && "${GRAPH_DIR_WAS_SET}" -eq 1 ]]; then
  GRAPH_SOURCE="${ORIGINAL_GRAPH_DIR}"
  GRAPH_SOURCE_FROM_ENV=1
fi
if [[ -z "${GRAPH_SOURCE}" ]]; then
  echo "[ramdisk] Unable to determine graph source directory. Set graph_source in ${CONFIG} or pass --graph-dir." >&2
  exit 1
fi

if [[ "${GRAPH_SOURCE}" != /* ]]; then
  GRAPH_SOURCE="$(python - "$GRAPH_SOURCE" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).resolve())
PY
)"
fi
if [[ "${GRAPH_SOURCE_FROM_ENV}" -eq 1 ]]; then
  echo "[ramdisk] Using graph source from existing GRAPH_DIR: ${GRAPH_SOURCE}"
fi

if [[ ! -d "${GRAPH_SOURCE}" ]]; then
  echo "[ramdisk] Graph source directory not found: ${GRAPH_SOURCE}" >&2
  exit 1
fi

BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS_OVERRIDE:-${CFG_BASELINE_NUM_WORKERS:-}}"
RAM_NUM_WORKERS="${RAM_NUM_WORKERS_OVERRIDE:-${CFG_RAM_NUM_WORKERS:-}}"

if [[ -n "${MANIFEST_OVERRIDE}" ]]; then
  export PIPELINE_MANIFEST="${MANIFEST_OVERRIDE}"
elif [[ "${PIPELINE_MANIFEST-__UNSET__}" == "__UNSET__" ]]; then
  # ensure downstream commands fall back to pipeline defaults
  unset PIPELINE_MANIFEST
fi

RAM_DEVICE=""
RAM_MOUNT=""
RAM_TEST_MODE=0
MANIFEST_MESSAGE_EMITTED=0

cleanup() {
  if [[ "${PIPELINE_MANIFEST_WAS_SET}" -eq 1 ]]; then
    export PIPELINE_MANIFEST="${ORIGINAL_PIPELINE_MANIFEST}"
  else
    unset PIPELINE_MANIFEST
  fi
  if [[ "${RAM_TEST_MODE}" -eq 0 && -n "${RAM_DEVICE}" ]]; then
    "${SCRIPT_DIR}/mac_mk_safe_ramdisk.sh" --detach "${RAM_DEVICE}" --force || true
  fi
}
trap cleanup EXIT INT TERM

baseline_duration=""
baseline_start_iso=""
baseline_end_iso=""
ram_duration=""
ram_start_iso=""
ram_end_iso=""

restore_num_workers() {
  if [[ -n "${1:-}" ]]; then
    export NUM_WORKERS_OVERRIDE="$1"
  else
    unset NUM_WORKERS_OVERRIDE
  fi
}

set_graph_dir() {
  if [[ -n "${1:-}" ]]; then
    export GRAPH_DIR="$1"
  else
    unset GRAPH_DIR
  fi
}

restore_graph_dir() {
  if [[ "${GRAPH_DIR_WAS_SET}" -eq 1 ]]; then
    export GRAPH_DIR="${ORIGINAL_GRAPH_DIR}"
  else
    unset GRAPH_DIR
  fi
}

run_pipeline() {
  if [[ "${MANIFEST_MESSAGE_EMITTED}" -eq 0 ]]; then
    if [[ -n "${PIPELINE_MANIFEST:-}" ]]; then
      echo "[ramdisk] Using manifest ${PIPELINE_MANIFEST}"
    else
      echo "[ramdisk] Using default manifest (run_full_pipeline.sh resolves manifests/run_core.yaml)."
    fi
    MANIFEST_MESSAGE_EMITTED=1
  fi
  if (( PIPELINE_HAS_ARGS )); then
    ./run_full_pipeline.sh "${PIPELINE_ARGS[@]}"
  else
    ./run_full_pipeline.sh
  fi
}

ORIGINAL_NUM_WORKERS="${NUM_WORKERS_OVERRIDE:-}"

if [[ "${SKIP_BASELINE}" -eq 0 ]]; then
  echo "[ramdisk] Starting baseline pipeline run (without RAM disk)..."
  if [[ -n "${BASELINE_NUM_WORKERS}" ]]; then
    export NUM_WORKERS_OVERRIDE="${BASELINE_NUM_WORKERS}"
  else
    restore_num_workers "${ORIGINAL_NUM_WORKERS}"
  fi
  set_graph_dir "${GRAPH_SOURCE}"
  baseline_start="$(date +%s)"
  baseline_start_iso="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  if run_pipeline; then
    baseline_end_iso="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    baseline_duration=$(( $(date +%s) - baseline_start ))
    echo "[ramdisk] Baseline run completed in ${baseline_duration}s."
  else
    echo "[ramdisk] Baseline run failed." >&2
    exit 1
  fi
fi

restore_graph_dir
restore_num_workers "${ORIGINAL_NUM_WORKERS}"

echo "[ramdisk] Provisioning RAM disk (${RAM_SIZE} with ${RAM_HEADROOM} headroom, name ${RAM_NAME})..."
set +e
create_output="$("${SCRIPT_DIR}/mac_mk_safe_ramdisk.sh" --size "${RAM_SIZE}" --headroom "${RAM_HEADROOM}" --name "${RAM_NAME}" 2>&1)"
create_status=$?
set -e
echo "${create_output}"

if [[ "${create_output}" == *"[TEST MODE]"* || "${SAFE_RAMDISK_TEST_MODE:-0}" != "0" || "${TEST_MODE:-0}" != "0" ]]; then
  RAM_TEST_MODE=1
  if [[ ${create_status} -ne 0 ]]; then
    echo "[ramdisk] mac_mk_safe_ramdisk.sh exited with status ${create_status} in test mode; continuing."
  fi
else
  if [[ ${create_status} -ne 0 ]]; then
    echo "[ramdisk] RAM disk provisioning failed:" >&2
    echo "${create_output}" >&2
    exit 1
  fi
fi

if [[ "${RAM_TEST_MODE}" -eq 1 ]]; then
  RAM_DEVICE=""
  RAM_MOUNT="${GRAPH_SOURCE}"
  echo "[ramdisk] mac_mk_safe_ramdisk.sh test mode detected; using source directory directly (${RAM_MOUNT})."
else
  RAM_DEVICE="$(echo "${create_output}" | awk '/Created:/ {print $2}')"
  RAM_MOUNT="$(echo "${create_output}" | awk '/Created:/ {print $NF}')"

  if [[ -z "${RAM_DEVICE}" || -z "${RAM_MOUNT}" || ! -d "${RAM_MOUNT}" ]]; then
    echo "[ramdisk] Failed to determine RAM disk mount point." >&2
    exit 1
  fi

  echo "[ramdisk] Syncing graph data from ${GRAPH_SOURCE} to ${RAM_MOUNT}..."
  rsync -a --delete "${GRAPH_SOURCE}/" "${RAM_MOUNT}/"
  echo "[ramdisk] Graph data staged at ${RAM_MOUNT}"
fi

set_graph_dir "${RAM_MOUNT}"
if [[ -n "${RAM_NUM_WORKERS}" ]]; then
  export NUM_WORKERS_OVERRIDE="${RAM_NUM_WORKERS}"
else
  restore_num_workers "${ORIGINAL_NUM_WORKERS}"
fi

echo "[ramdisk] Starting RAM-disk pipeline run..."
ram_start="$(date +%s)"
ram_start_iso="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
if run_pipeline; then
  ram_end_iso="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  ram_duration=$(( $(date +%s) - ram_start ))
  echo "[ramdisk] RAM-disk run completed in ${ram_duration}s."
else
  echo "[ramdisk] RAM-disk run failed." >&2
  exit 1
fi

restore_graph_dir
restore_num_workers "${ORIGINAL_NUM_WORKERS}"

benchmark_dir="${SCRIPT_DIR}/training_runs/ramdisk_benchmarks"
mkdir -p "${benchmark_dir}"

PIPELINE_ARGS_STRING=""
if (( PIPELINE_HAS_ARGS )); then
  PIPELINE_ARGS_STRING="${PIPELINE_ARGS[*]}"
fi

python - "$benchmark_dir" "${baseline_duration:-}" "${ram_duration:-}" "${GRAPH_SOURCE}" "${RAM_MOUNT}" "${RAM_DEVICE}" "${RAM_SIZE}" "${RAM_HEADROOM}" "${RAM_NAME}" "${baseline_start_iso:-}" "${baseline_end_iso:-}" "${ram_start_iso}" "${ram_end_iso}" "${RAM_TEST_MODE}" "${PIPELINE_ARGS_STRING}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

benchmark_dir = Path(sys.argv[1])
baseline_duration = sys.argv[2]
ram_duration = sys.argv[3]
graph_source = sys.argv[4]
ram_mount = sys.argv[5]
ram_device = sys.argv[6]
ram_size = sys.argv[7]
ram_headroom = sys.argv[8]
ram_name = sys.argv[9]
baseline_start_iso = sys.argv[10] or None
baseline_end_iso = sys.argv[11] or None
ram_start_iso = sys.argv[12]
ram_end_iso = sys.argv[13]
test_mode = sys.argv[14]
pipeline_args = sys.argv[15]

def parse_duration(value):
    value = value.strip()
    if not value:
        return None
    return int(value)

now = datetime.now(timezone.utc)
is_test_mode = bool(int(test_mode)) if test_mode and test_mode.strip() else False
parsed_pipeline_args = pipeline_args.split() if pipeline_args and pipeline_args.strip() else []

record = {
    "timestamp": now.isoformat(),
    "ramdisk": {
        "name": ram_name,
        "size": ram_size,
        "headroom": ram_headroom,
        "mount_point": ram_mount,
        "device": ram_device,
        "test_mode": is_test_mode,
    },
    "graph_source": graph_source,
    "pipeline_args": parsed_pipeline_args,
    "baseline_seconds": parse_duration(baseline_duration),
    "ram_seconds": parse_duration(ram_duration),
    "baseline_start": baseline_start_iso,
    "baseline_end": baseline_end_iso,
    "ram_start": ram_start_iso,
    "ram_end": ram_end_iso,
}

output = benchmark_dir / f"ramdisk_benchmark_{now.strftime('%Y%m%d_%H%M%S')}.json"
with output.open("w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2)

print(f"[ramdisk] Benchmark written to {output}")
PY

echo "[ramdisk] All runs complete. RAM disk detached."
