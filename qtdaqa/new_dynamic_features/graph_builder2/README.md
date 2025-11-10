# Dynamic Graph Builder

This directory contains the feature-extraction engine for the dynamic TopoQA
pipeline. For each PDB structure it runs four pluggable stages:

1. **Interface** – identify interface residues and coordinates.
2. **Topology** – compute persistent-homology descriptors around each residue.
3. **Node** – merge DSSP-derived features with topology statistics.
4. **Edge** – connect residues across chains and emit edge attributes.

Outputs are PyTorch Geometric graphs (`*.pt`) plus metadata files that model
training and inference rely on (`graph_metadata.json`,
`graph_builder_summary.json`). Everything is driven by a user-supplied
`feature-config.yaml`.

---

## Quick Start

```bash
cd qtdaqa/new_dynamic_features/graph_builder2
./run_graph_builder2.sh \
  --dataset-dir /path/to/PDBs \
  --work-dir /tmp/graph_builder/work \
  --graph-dir /tmp/graph_builder/graph_data \
  --log-dir /tmp/graph_builder/logs \
  --jobs 12 \
  --feature-config /abs/path/feature-config.yaml
```

All five path flags are **mandatory**; the builder will exit immediately if any
is missing or if the feature config cannot be read. You can also run the Python
module directly:

```bash
python -m qtdaqa.new_dynamic_features.graph_builder2.graph_builder2 ...
```

Helpers:

- `./run_graph_builder2.sh --list-modules` – list every registered interface,
  topology, node, edge, and custom module kind with summaries and parameter descriptions. Add
  `--list-modules-format markdown` to emit README-ready Markdown.
- `./run_graph_builder2.sh --create-feature-config` – write a minimal template
  with inline instructions (no giant catalog to edit by hand).

---

## Feature Configuration

`feature-config.yaml` selects one module per stage and optionally sets global
defaults. The builder validates the schema up front (missing sections or
malformed parameters cause the run to stop immediately).

```yaml
defaults:
  jobs: 12          # optional global override for stage workers

interface:
  module: interface/polar_cutoff/v1
  params:
    cutoff: 14.0
    coordinate_decimals: 3

topology:           # optional, but common
  module: topology/persistence_basic/v1
  params:
    neighbor_distance: 8.0
    filtration_cutoff: 8.0
    min_persistence: 0.01
    element_filters:
      - ['C']
      - ['N']
      - ['O']
      - ['C', 'N']
      - ['C', 'O']
      - ['N', 'O']
      - ['C', 'N', 'O']

node:
  module: node/dssp_topo_merge/v1
  params:
    drop_na: false

edge:
  module: edge/multi_scale/v24
  params:
    histogram_bins: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    contact_threshold: 5.0

# Optional extra stage (custom modules):
# mol:
#   module: custom/mol_stage/v1
#   params: {}
```

Rules:

- `interface`, `node`, and `edge` sections are required.
- Additional sections (e.g., `topology`, `mol`) are allowed as long as they have
  the same `module`/`params` structure.
- Common mistakes (like writing tuple literals for `element_filters`) are caught
  before the builder starts processing PDBs.

To switch to the legacy 11‑D edge features, change the `edge` block to
`edge/legacy_band/v11` with the desired distance window.

### Edge Modules

- `edge/multi_scale/v24` – default 24‑D geometric features (distance, inverse, unit
  vectors, chain/residue encodings, multi-band histograms, single contact count).
- `edge/legacy_band/v11` – 11‑D histogram mirroring the historical TopoQA
  training data (0–10 Å pairs, 1 Å histogram bins, MinMax scaling).
- `edge/neo/v24` – hybrid module combining the proven 11‑D histogram window with
  multi-scale geometry. It supports 0–10 Å truncated bins, multiple contact
  thresholds, optional long-band masks, and lightweight feature scaling. Select it
  when you want the geometric expressiveness of multi-scale edges without giving up
  the DockQ-friendly histogram behavior.

---

## Outputs & Metadata

For each run you’ll find:

- `graph_dir/*.pt` – PyTorch Geometric graphs. Each `data` object includes
  `data.metadata` describing the edge module, parameters, and sample counts.
- `graph_dir/graph_metadata.json` – canonical schema used by model training and
  inference to learn the node/edge layouts (dimensions, module IDs, node column
  names, sample graphs, etc.).
- `graph_dir/schema_summary.json` – human-readable snapshot of the resolved
  schema (module registry, node/edge dims, notes). Useful for quick inspection
  or when debugging legacy checkpoints.
- `log_dir/graph_builder_summary.json` – run summary (module selections, job
  counts, success/failure tallies) plus stage logs for debugging.

Keep these metadata files with your graphs. Model training automatically loads
`graph_metadata.json`, and model inference regenerates graphs from the metadata
embedded in training checkpoints.

---

## Troubleshooting

- **“Feature configuration not found…”** – the builder now requires
  `--feature-config`; double-check the absolute path you passed.
- **Schema errors at startup** – the YAML is missing a required section or has a
  malformed parameter (e.g., `element_filters` not written as a YAML list). Fix
  the config and rerun; the builder won’t process any PDBs until the config is
  valid.
- **Need to inspect modules/params** – use `--list-modules`. Each entry lists
  the default parameters and descriptions so you can craft custom configs
  without digging through the source.

---

## Regenerating the Module Catalog

To keep this README aligned with the current registry, generate the Markdown
module catalog with:

```bash
./run_graph_builder2.sh --list-modules --list-modules-format markdown > module_catalog.md
```

Review the resulting file and copy the updated sections into the README whenever
modules or defaults change.
