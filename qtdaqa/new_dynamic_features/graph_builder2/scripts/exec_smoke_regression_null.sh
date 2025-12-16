#!/usr/bin/env bash
set -euo pipefail

# Smoke regression for topology/persistence_null/v1 (ablation control).
# Runs on a tiny dataset and asserts metadata dims/schemas to ensure compatibility.
# Usage: ./exec_smoke_regression_null.sh [dataset_dir] [jobs]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

DATASET_DIR="${1:-${REPO_ROOT}/datasets/training/adjusted/smaller_pilot_batch_Dockground_MAF2}"
JOBS="${2:-4}"

CFG_PATH="${SCRIPT_DIR}/../feature_configs/null_topology/feature-config.yaml.null.10A"
OUTPUT_DIR="${SCRIPT_DIR}/output/smoke_null"
WORK_DIR="${OUTPUT_DIR}/work"
GRAPH_DIR="${OUTPUT_DIR}/graph_data"
LOG_DIR="${OUTPUT_DIR}/logs"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${WORK_DIR}" "${GRAPH_DIR}"

echo "== Running null topology smoke with config ${CFG_PATH} ==" >&2
time "${SCRIPT_DIR}/../run_graph_builder2.sh" --pdb-warnings --no-sort-artifacts \
  --dataset-dir "${DATASET_DIR}" \
  --work-dir "${WORK_DIR}" \
  --graph-dir "${GRAPH_DIR}" \
  --log-dir "${LOG_DIR}" \
  --jobs "${JOBS}" \
  --feature-config "${CFG_PATH}"

python - "${GRAPH_DIR}" <<'PY'
import json
from pathlib import Path
import sys

graph_dir = Path(sys.argv[1])
expected_topo = 140
expected_node = 172
expected_edge = 11

topo_cols_path = graph_dir / "topology_columns.json"
if not topo_cols_path.exists():
    raise SystemExit(f"topology_columns.json missing at {topo_cols_path}")

meta_path = graph_dir / "graph_metadata.json"
if not meta_path.exists():
    raise SystemExit(f"graph_metadata.json missing at {meta_path}")
meta = json.loads(meta_path.read_text())

errors = []
topo_dim = meta.get("topology_feature_dim")
node_dim = meta.get("node_feature_dim")
edge_dim = meta.get("edge_feature_dim")
if topo_dim != expected_topo:
    errors.append(f"topology_feature_dim={topo_dim} != expected {expected_topo}")
if node_dim != expected_node:
    errors.append(f"node_feature_dim={node_dim} != expected {expected_node}")
if edge_dim != expected_edge:
    errors.append(f"edge_feature_dim={edge_dim} != expected {expected_edge}")

topo_schema = meta.get("_topology_schema") or {}
if not topo_schema.get("columns"):
    errors.append("missing topology columns in schema")
if topo_schema.get("dim") != expected_topo:
    errors.append(f"topology_schema.dim={topo_schema.get('dim')} != {expected_topo}")

if errors:
    for e in errors:
        print(e, file=sys.stderr)
    raise SystemExit(1)
print("Null topology smoke: metadata dims/schemas OK")
PY

echo "Null topology smoke regression passed."
