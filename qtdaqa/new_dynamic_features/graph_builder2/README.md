OUTPUT="output/11d_topoqa_10A"
FEATURE_FILE="feature-config.yaml.topoqa.10A"

# mk dirs
mkdir -p ${OUTPUT}/work
mkdir -p ${OUTPUT}/graph_data

time ./run_graph_builder2.sh --pdb-warnings \
  --dataset-dir datasets/training/adjusted/Dockground_MAF2 \
  --work-dir qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/work \
  --graph-dir qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data \
  --log-dir qtdaqa/new_dynamic_features/graph_builder2/logs_${OUTPUT} \
  --jobs 16 \
  --feature-config qtdaqa/new_dynamic_features/graph_builder2/${FEATURE_FILE}

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
  with inline instructions (no giant catalog to edit by hand). Add
  `--include-alternates` if you also want fully-detailed (commented) blocks for every
  registered module so you can copy/paste parameters.

Edge CSV dumps are **enabled by default**. They land under `<work-dir>/edge_features`.
Pass `--edge-dump-dir /abs/elsewhere` to redirect them, `--no-dump-edges` to skip
generating them, or `--dump-edges` to force them back on even if the feature config
disables dumping.

Determinism:
- Interface files are always sorted (chain → residue_seq → insertion_code → residue_name → x → y → z).
- Edge ordering inside `.pt` files is always sorted by (src_idx, dst_idx, distance).
- Additional “belt-and-suspenders” sorting for topology/node/edge CSV artifacts is **on by default**; disable with `--no-sort-artifacts`. Disabling does not affect the mandatory interface or in-graph edge ordering.

---

## Feature Configuration

`feature-config.yaml` selects one module per stage and optionally sets global
defaults. The builder validates the schema up front (missing sections or
malformed parameters cause the run to stop immediately).

```yaml
defaults:
  jobs: 16          # optional global override for stage workers

interface:
  module: interface/polar_cutoff/v1  # alias: TopoQA default 10A cut-off
  alias: "TopoQA default 10A cut-off"
  summary: "Polar interface detection using distance cutoff."
  description: "Identifies interface residues across chains using the polar interface routine (new_calculate_interface.process_pdb_file) with configurable distance cutoff and coordinate rounding."
  params:
    cutoff: 10.0
    coordinate_decimals: -1  # skip rounding to keep raw coords
    jobs: 16

topology:           # optional, but common
  module: topology/persistence_basic/v1  # alias: TopoQA default
  alias: "TopoQA default"
  summary: "Persistent homology features with configurable distance thresholds."
  description: "Generates per-residue topological descriptors using the persistent homology pipeline from new_topological_features. Interface residues are required as input."
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
    jobs: 16

node:
  module: node/dssp_topo_merge/v1  # alias: TopoQA default
  alias: "TopoQA default"
  summary: "DSSP-derived node features merged with topology statistics."
  description: "Runs the DSSP-based node feature extractor (node_fea) and merges results with persistent homology statistics aligned by residue IDs."
  params:
    drop_na: true  # matches fea_df_clean = fea_df.dropna() in both inference_model.py and k_mac_inference_pca_tsne4.py
    jobs: 16

edge:
  module: edge/legacy_band/v11
  alias: ""
  summary: "11-D legacy edge features (distance + atom distance histogram)."
  description: "Reproduces the original 11-dimensional edge attributes consisting of residue pair distance and a 10-bin histogram of inter-atomic distances, scaled to [0, 1]."
  params:
    distance_min: 0.0
    distance_max: 10.0
    scale_features: true
    jobs: 16

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
- Edge dumps can also be toggled inside the YAML via `options.edge_dump: false` and
  redirected with `options.edge_dump_dir: /abs/path`, but CLI flags (`--dump-edges`,
  `--no-dump-edges`, `--edge-dump-dir`) take precedence when provided.

To switch to the 24‑D multi-scale edges, change the `edge` block to
`edge/multi_scale/v24` and configure its distance bands / histogram bins.

### Edge Modules

- `edge/legacy_band/v11` – default 11‑D histogram mirroring the historical TopoQA
  training data (0–10 Å pairs, 1 Å histogram bins, MinMax scaling).
- `edge/multi_scale/v24` – 24‑D geometric features (distance, inverse, unit
  vectors, chain/residue encodings, multi-band histograms, single contact count).
- `edge/neo/v24` – hybrid module combining the proven 11‑D histogram window with
  multi-scale geometry. It supports 0–10 Å truncated bins, multiple contact
  thresholds, optional long-band masks, and lightweight feature scaling. Select it
  when you want the geometric expressiveness of multi-scale edges without giving up
  the DockQ-friendly histogram behavior.
- `edge/legacy_plus_topo_pair` – builds on the 11‑D legacy module and appends
  20 persistent-homology statistics per edge. By default it considers the two
  residues forming the edge plus neighbors within 4 Å, runs a small Rips complex,
  and concatenates the topology summary to the histogram vector. Tune
  `neighbor_distance`, `include_neighbors`, `filtration_cutoff`, and
  `min_persistence` to control the geometric context captured for each pair.
- `edge/edge_plus_bal_agg_topo/v1` (lean) – legacy 11‑D histogram prepended to balanced
  aggregation of endpoint topology: concat(u,v), mean, abs-diff, cosine, and norms
  (no min/max).
- `edge/edge_plus_min_agg_topo/v1` (lean) – prepends the legacy 11‑D histogram
  (distance + 10-bin atom histogram) and concatenates per-residue topology for each
  endpoint: concat(u_topo, v_topo), abs-diff, cosine similarity, and norms
  (defaults on). Deterministic edge ordering is preserved; histogram block can be
  scaled independently via `scale_histogram`.
- (Planned) Aggregated-topology edge variants – reuse the per-residue topology
  vectors already computed for interface residues to build relational signals per
  edge without rerunning persistence. For an edge (u, v), combine the endpoint
  topo vectors with:
  * Asymmetric context: keep both endpoints (concat) so source/target can differ.
  * Symmetric relations: mean, abs-diff (and optionally min/max, cosine) to show
    how similar or different the endpoints’ topology is.
  * Optional local context: pool neighbors’ topo around each endpoint, then apply
    the same mean/abs-diff (optionally min/max/cosine) to the pooled summaries.
  This augments the legacy 11‑D histogram with cheap, deterministic relational
  topology, leveraging the topo features that already performed well at the node
  level. These will preserve the existing deterministic edge ordering
  (src_idx, dst_idx, distance) and rely on the canonical interface/node/topology
  sorting already in place.

### Planned aggregated-topology edge modules (defaults and rationale)

These three module families are planned (edge_plus_min_agg_topo lean is now implemented as
`edge/edge_plus_min_agg_topo/v1`). Defaults below are meant to be sensible starting points
that balance signal, cost, and determinism:

- `edge_plus_min_agg_topo`
  - Lean defaults: `include_norms=True`, `include_cosine=True`; no min/max block.
  - Heavy defaults: `include_minmax=True`, `include_norms=True`, `include_cosine=True`.
  - Rationale: the lean form keeps footprint small while still conveying relative
    magnitude (norms) and directional similarity (cosine) between endpoint topo
    vectors; the heavy form adds symmetric min/max for fuller contrast.

- `edge_plus_bal_agg_topo`
  - Lean defaults: include endpoint `mean`, `abs_diff`, `cosine`, and `norms`
    (omit min/max).
  - Heavy defaults: same as lean plus `min/max` on endpoints.
  - Rationale: balanced summary (concat + mean + abs-diff) already captures
    shared and divergent structure; cosine and norms add low-cost orientation and
    scale cues. Heavy adds min/max to expose extremal differences without changing
    ordering or determinism.

- `edge_plus_pool_agg_topo`
  - Shared defaults: neighbor pooling `k=5` (midpoint of 4–6) for deterministic,
    modest local context.
  - Lean defaults: endpoint block uses the balanced-lean set (mean/abs-diff/norms/
    cosine, no min/max); pooled block uses mean/abs-diff/norms/cosine, no min/max.
  - Heavy defaults: add min/max to both endpoint and pooled blocks while keeping
    norms and cosine; keep `k=5`.
  - Rationale: pooling brings nearby topo context without rerunning persistence;
    k=5 keeps compute predictable. Lean avoids feature blow-up while leveraging
    norms/cosine for stability; heavy parallels other heavy variants with min/max
    on both raw and pooled summaries.

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
