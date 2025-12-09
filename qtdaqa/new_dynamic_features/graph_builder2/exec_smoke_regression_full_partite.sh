#!/usr/bin/env bash
set -euo pipefail

# Smoke regression for topology/persistence_full_k_partite/v1.
# Runs minimal/lean/heavy/heavy_stratified on a tiny dataset and asserts metadata dims/schemas.
# Usage: ./exec_smoke_regression_full_partite.sh [dataset_dir] [jobs]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_DIR="${1:-${REPO_ROOT}/datasets/training/adjusted/smaller_pilot_batch_Dockground_MAF2}"
JOBS="${2:-4}"

run_smoke() {
  local preset_name="$1"
  local cfg_path="$2"
  local output_dir="${SCRIPT_DIR}/output/smoke_full_${preset_name}"
  local work_dir="${output_dir}/work"
  local graph_dir="${output_dir}/graph_data"
  local log_dir="${output_dir}/logs"

  rm -rf "${output_dir}"
  mkdir -p "${work_dir}" "${graph_dir}"

  echo "== Running full_k_partite preset ${preset_name} with config ${cfg_path} ==" >&2
  time "${SCRIPT_DIR}/run_graph_builder2.sh" --pdb-warnings --no-sort-artifacts \
    --dataset-dir "${DATASET_DIR}" \
    --work-dir "${work_dir}" \
    --graph-dir "${graph_dir}" \
    --log-dir "${log_dir}" \
    --jobs "${JOBS}" \
    --feature-config "${cfg_path}"

python - "${graph_dir}" "${preset_name}" <<'PY'
import json
from pathlib import Path
import sys

graph_dir = Path(sys.argv[1].rstrip("}"))  # accommodate accidental brace in path
preset = sys.argv[2]

expected_topo = {"minimal": 140, "lean": 560, "heavy": 700, "heavy_stratified": 900}.get(preset)
expected_node = 172
expected_edge = 11

topo_cols_path = graph_dir / "topology_columns.json"
if not topo_cols_path.exists():
    raise SystemExit(f"{preset}: topology_columns.json missing at {topo_cols_path}")

meta_path = graph_dir / "graph_metadata.json"
if not meta_path.exists():
    raise SystemExit(f"{preset}: graph_metadata.json missing at {meta_path}")
meta = json.loads(meta_path.read_text())

topo_dim = meta.get("topology_feature_dim")
node_dim = meta.get("node_feature_dim")
edge_dim = meta.get("edge_feature_dim")

errors = []
if topo_dim != expected_topo:
    errors.append(f"{preset}: topology_feature_dim={topo_dim} != expected {expected_topo}")
if node_dim != expected_node:
    errors.append(f"{preset}: node_feature_dim={node_dim} != expected {expected_node}")
if edge_dim != expected_edge:
    errors.append(f"{preset}: edge_feature_dim={edge_dim} != expected {expected_edge}")

topo_schema = meta.get("_topology_schema") or {}
node_schema = meta.get("_node_schema") or {}
if not topo_schema.get("columns"):
    errors.append(f"{preset}: missing topology columns")
if topo_schema.get("dim") != expected_topo:
    errors.append(f"{preset}: topology_schema.dim={topo_schema.get('dim')} != {expected_topo}")
if not node_schema.get("columns"):
    errors.append(f"{preset}: missing node columns")
if node_schema.get("dim") != expected_node:
    errors.append(f"{preset}: node_schema.dim={node_schema.get('dim')} != {expected_node}")
if node_schema.get("topology_dim") != expected_topo:
    errors.append(f"{preset}: node_schema.topology_dim={node_schema.get('topology_dim')} != {expected_topo}")

if errors:
    for e in errors:
        print(e, file=sys.stderr)
    raise SystemExit(1)
print(f"{preset}: metadata dims/schemas OK")
PY
}

run_smoke "minimal"         "${SCRIPT_DIR}/feature_configs/feature-config.yaml.full_minimal.10A"
run_smoke "lean"            "${SCRIPT_DIR}/feature_configs/feature-config.yaml.full_lean.10A"
run_smoke "heavy"           "${SCRIPT_DIR}/feature_configs/feature-config.yaml.full_heavy.10A"
run_smoke "heavy_stratified" "${SCRIPT_DIR}/feature_configs/feature-config.yaml.full_heavy_stratified.10A"

echo "All full_k_partite smoke regressions passed."
