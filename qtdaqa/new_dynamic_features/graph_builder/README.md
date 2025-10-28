# Dynamic Graph Builder (modular)

This copy of the graph builder implements a fully modular pipeline where every
feature stage (interface, topology, node, edge) is provided by a pluggable
module. Feature selection, thresholds, and per-stage options are controlled by
`features.yaml`, making it simple to switch between the current 24‑D edge
features and the legacy 11‑D variant without touching code.

## Quick start

```bash
python qtdaqa/new_dynamic_features/graph_builder/graph_builder.py \
  --dataset-dir datasets/training/adjusted/Dockground_MAF2 \
  --work-dir runs/dynamic_work \
  --graph-dir runs/dynamic_graphs \
  --log-dir runs/dynamic_logs \
  --jobs 8
```

The run creates a timestamped folder under `runs/dynamic_logs`, fills
`runs/dynamic_work` with intermediate CSVs, and writes `.pt` graphs to
`runs/dynamic_graphs`. A machine-readable summary (including active modules and
parameters) is emitted as `graph_builder_summary.json`.

## Feature configuration (`features.yaml`)

```yaml
interface:
  module: interface/polar_cutoff/v1
  params:
    cutoff: 14.0
    coordinate_decimals: 3

topology:
  module: topology/persistence_basic/v1
  params:
    neighbor_distance: 8.0
    filtration_cutoff: 8.0
    min_persistence: 0.01

node:
  module: node/dssp_topo_merge/v1
  params:
    drop_na: false

edge:
  module: edge/multi_scale/v24
  params:
    histogram_bins: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    contact_threshold: 5.0

options:
  edge_dump: false
```

Switch to the legacy 11‑D edges by changing the edge block:

```yaml
edge:
  module: edge/legacy_band/v11
  params:
    distance_min: 0.0
    distance_max: 10.0
```

Place `features.yaml` in your work directory to override the defaults, or supply
`--feature-config /path/to/features.yaml` on the CLI.

## Discovering installed modules

```bash
python qtdaqa/new_dynamic_features/graph_builder/graph_builder.py --list-modules
```

This prints all registered interface/topology/node/edge modules along with a
short description and their configurable parameters. New modules can be added
under `modules/` by subclassing the appropriate base class and calling
`register_feature_module`.

## Stage outputs

| Stage | Module key | Output |
| --- | --- | --- |
| Interface | `interface/*` | `work_dir/interface/*.interface.txt` and per-PDB logs |
| Topology | `topology/*` | `work_dir/topology/*.topology.csv` |
| Node | `node/*` | `work_dir/node_features/*.csv` |
| Edge | `edge/*` | `.pt` graphs in `graph_dir`, metadata sidecar, optional edge dumps |

Each `.pt` file embeds a `data.metadata` dictionary capturing the edge module
ID, the resolved parameters, and any edge-specific metadata. The run summary
JSON records the same information alongside per-stage success counts, making it
easy for downstream training or inference code to validate schema compatibility.
