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
  `--list-modules-format markdown` to emit README-ready Markdown, or `json` for scripting.
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

# Alternate topology example (hybrid PH + Laplacian)
# topology:
#   module: topology/persistence_laplacian_hybrid/v1
#   alias: "140D topology + 32D Laplacian"
#   summary: "Persistent homology plus Laplacian spectral moments per interface residue."
#   params:
#     neighbor_distance: 8.0
#     filtration_cutoff: 8.0
#     min_persistence: 0.01
#     element_filters:
#       - [C]
#       - [N]
#       - [O]
#       - [C, N]
#       - [C, O]
#       - [N, O]
#       - [C, N, O]
#     lap_graph_mode: cross_chain
#     lap_distance_cutoff: 8.0
#     lap_edge_weight: gaussian
#     lap_sigma: 4.0
#     lap_eigs_count: 16
#     lap_moment_orders: [1, 2, 3, 4]
#     lap_heat_times: [0.1, 1.0, 5.0]

### Bipartite interface graph (TopoQA definition)
This is the graph a Laplacian-only or Laplacian-augmented topology module operates on:

```
1) Identify interface residues:
   For each residue i in chain A:
     if any atom of i is within cutoff (e.g., 10 A) of any atom in chain B,
     then i is in U (interface residues of chain A).
   Repeat for chain B -> V.

2) Build bipartite graph G = (U ∪ V, E):
   Nodes: U (chain A interface residues) + V (chain B interface residues)
   Edges: (u, v) in E iff u in U, v in V, and dist(Ca_u, Ca_v) < cutoff
   No A-A or B-B edges.

3) Generalize to k-partite (k > 2):
   Partition nodes by chain/label (P1..Pk).
   Add edges only between different partitions (Pi <-> Pj, i != j).

4) Compute Laplacian features on G:
   Build adjacency (weighted or unweighted), form Laplacian L,
   extract eigenvalue stats/moments/heat traces as configured.
```

# Null topology (ablation control; keeps 140D shape but no signal)
# topology:
#   module: topology/persistence_null/v1
#   alias: "Null topology (constant zeros) = 140D ablation"
#   summary: "Topology ablation: constant features with PH-compatible schema."
#   params:
#     element_filters:
#       - [C]
#       - [N]
#       - [O]
#       - [C, N]
#       - [C, O]
#       - [N, O]
#       - [C, N, O]
#     constant_value: 0.0
#     jobs: 16

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

# Lean Laplacian moments (MoL) modules (new):
# topology:
#   module: topology/lightweight_MoL/v1  # alias: 140D PH + 8D unweighted Lap moments (mu1-4, kappa2-4) = Topology 148D (lean MoL)
#   params:
#     neighbor_distance: 8.0
#     filtration_cutoff: 8.0
#     min_persistence: 0.01
#     dedup_sort: false
#     element_filters:
#       - [C]
#       - [N]
#       - [O]
#       - [C, N]
#       - [C, O]
#       - [N, O]
#       - [C, N, O]
#     lap_k_neighbors: null
#     lap_max_neighbors: 128
#     lap_size_threshold: 80    # exact eigs when nodes <= threshold, else SLQ
#     lap_estimator: exact      # exact or slq
#     lap_slq_probes: 8
#     lap_slq_steps: 32
#     lap_profile: false
#     jobs: 16
# topology (PH-free Laplacian-only replacement):
#   module: topology/standalone_MoL_replace_topology/v1  # alias: Laplacian-only spectral topology (32 eigs, entropy+Fiedler+Kirchhoff, multi-scale 6/8/10 Å)
#   params:
#     neighbor_distance: 8.0        # single-scale when lap_multi_radii=null
#     lap_multi_radii: [6.0, 8.0, 10.0]  # default multi-scale; set null for single-scale
#     lap_weight: unweighted        # or gaussian (sigma=cutoff/2), inverse
#     lap_normalize: sym            # or rw
#     lap_eigs_count: 32            # 24 for leaner/faster
#     lap_heat_times: [0.1, 1.0, 5.0]
#     lap_include_entropy: true
#     lap_include_fiedler: true
#     lap_include_kirchhoff: true
#     lap_use_centered_moments: true
#     lap_k_neighbors: null
#     lap_max_neighbors: 128
#     lap_estimator: exact
#     lap_size_threshold: 80
#     lap_slq_probes: 8
#     lap_slq_steps: 32
#     jobs: 16
# edge:
#   module: edge/edge_plus_lightweight_MoL/v1  # alias: Legacy 11D Edge + 5D unweighted Lap moments (mu1-3, kappa2, kappa3) on pair neighborhood = Edge 16D (lean MoL)
#   params:
#     distance_min: 0.0
#     distance_max: 10.0
#     scale_histogram: true
#     lap_size_threshold: 80   # exact eigs when nodes <= threshold, else SLQ
#     lap_estimator: exact     # exact or slq
#     lap_slq_probes: 8
#     lap_slq_steps: 32
#     lap_max_neighbors: 128
#     lap_profile: false
#     jobs: 16

# Example pairing with hybrid topology:
# topology:
#   module: topology/persistence_laplacian_hybrid/v1
#   params: {...}
# edge:
#   module: edge/edge_plus_min_agg_lap_hybrid/v1  # Use with 172D PH+Lap topology vectors
#   params:
#     distance_min: 0.0
#     distance_max: 10.0
#     scale_histogram: true
#     include_norms: true
#     include_cosine: true
#     include_minmax: false
#     variant: lean
# edge:
#   module: edge/edge_plus_bal_agg_lap_hybrid/v1  # or edge_plus_pool_agg_lap_hybrid/v1 for pooled variant
#   params:
#     distance_min: 0.0
#     distance_max: 10.0
#     scale_histogram: true
#     include_norms: true
#     include_cosine: true
#     include_minmax: false
#     variant: lean
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

### Node Modules

- `node/dssp_topo_merge/v1` – DSSP-derived node features merged with topology statistics (assumes legacy 140D topology).
- `node/dssp_topo_merge_passthrough/v1` – DSSP/basic 32D features merged with all topology columns (passthrough; dynamic dims, preserves PH+Lap).

### Edge Modules

- `edge/legacy_band/v11` – 11‑D legacy histogram (distance + 10-bin atom histogram).
- `edge/multi_scale/v24` – 24‑D geometric (distance/inverse/unit vectors/encodings/histograms/contact count).
- `edge/edge_plus_min_agg_topo/v1` – legacy 11‑D histogram + concat/abs-diff/cosine/norms of endpoint topo (lean); heavy adds per-dimension min/max.
- `edge/edge_plus_bal_agg_topo/v1` – legacy 11‑D histogram + balanced topo agg (concat + mean + abs-diff + cosine + norms); heavy adds min/max.
- `edge/edge_plus_pool_agg_topo/v1` – legacy 11‑D histogram + balanced topo agg + pooled neighbor topo means; heavy adds min/max to endpoint and pooled summaries.
- `edge/edge_plus_min_agg_lap_hybrid/v1` – same as edge_plus_min_agg_topo but documented for hybrid topology (PH+Lap 172D).
- `edge/edge_plus_bal_agg_lap_hybrid/v1` – same as edge_plus_bal_agg_topo but documented for hybrid topology (PH+Lap 172D).
- `edge/edge_plus_pool_agg_lap_hybrid/v1` – same as edge_plus_pool_agg_topo but documented for hybrid topology (PH+Lap 172D).
- Deregistered (retained for reference only; not auto-loaded): `edge/multi_scale_v24`, `edge/neo_v24`, `edge/legacy_plus_topo_pair` (moved to `modules/edge/deregistered`).

Use `--list-modules --list-modules-format json|markdown` to see full parameter/default listings for every module.
`--create-feature-config --include-alternates` now emits a `# dim:` comment next to aliases when a dimension hint is known (variant-aware); dynamic passthrough aliases suppress the hint.

---

## Outputs & Metadata

For each run you’ll find:

- `graph_dir/*.pt` – PyTorch Geometric graphs. Each `data` object includes
  `data.metadata` describing the edge module, parameters, and sample counts.
- `graph_dir/graph_metadata.json` – canonical schema used by model training and
  inference to learn the node/edge layouts (dimensions, module IDs, node column
  names, sample graphs, etc.). `node_feature_columns` and `node_feature_dim`
  are populated for downstream consumers.
- `graph_dir/schema_summary.json` – human-readable snapshot of the resolved
  schema (module registry, node/edge dims, notes). Topology columns (auto-extracted
  during the edge stage when topology CSVs are present) and node_feature_columns
  are included when available; node_feature_dim is mirrored when present.
  Useful for quick inspection or when debugging legacy checkpoints.

### Feature dimensionality (defaults)
- Topology: `topology/persistence_basic/v1` → 140 dims; `topology/persistence_laplacian_hybrid/v1` → 172 dims (140 PH + 32 Laplacian).
- Node: `node/dssp_topo_merge/v1` → 172 dims (32 DSSP/basic + 140 topology). `node/dssp_topo_merge_passthrough/v1` → 32 DSSP/basic + all topology columns (dynamic; e.g., 32 + 172 = 204 with hybrid topology).
- Edge (with hybrid topology topo_dim=172):
  - `edge_plus_min_agg_lap_hybrid` lean: 530 dims (11 hist + 3×172 + norms+cosine); heavy: 874 dims (+min/max).
  - `edge_plus_bal_agg_lap_hybrid` lean: 702 dims (11 hist + 4×172 + norms+cosine); heavy: 1,046 dims (+min/max).
  - `edge_plus_pool_agg_lap_hybrid` lean: 1,393 dims (11 hist + endpoint agg + pooled agg); heavy: 2,081 dims (+min/max in both blocks).
- Laplacian-only topology (no PH): `topology/standalone_MoL_replace_topology/v1` → 55 dims single-scale (32 eigs, entropy, centered moments, Fiedler, Kirchhoff); multi-scale default [6/8/10 Å] → 165 dims (3× block). 24 eigs is a leaner/faster variant.
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
- **Laplacian modules (performance/guards)** – neighborhoods are capped by
  `lap_max_neighbors`; size_threshold controls exact→SLQ; degenerate/empty
  neighborhoods zero-fill. Enable `lap_profile` to log per-block timing if
  needed.

---

## Regenerating the Module Catalog

To keep this README aligned with the current registry, generate the Markdown
module catalog with:

```bash
./run_graph_builder2.sh --list-modules --list-modules-format markdown > module_catalog.md
```

Review the resulting file and copy the updated sections into the README whenever
modules or defaults change.

---

## Guidance for adding new modules

- Keep outputs deterministic: sort rows when artifact sorting is enabled, and use lexicographic tie-breakers for any neighbor selection.
- Provide sensible defaults, aliases, and clear summaries/descriptions so `--list-modules` and `--create-feature-config` remain informative.
- Validate parameters and keep them YAML-friendly; expose param comments via `config_template`. Include `# dim` hints and alternates so `--include-alternates` output stays readable.
- Record feature dimensions in metadata and keep CSV outputs ID-first; ensure new modules appear in listings/templates and in `schema_summary.json`/`graph_metadata.json`.
- Edge dumps: honor `dump_path`; write src/dst/distance; `edge_runner` will sort dumps when enabled.
- Sorting: interface/edge ordering is always deterministic; `--no-sort-artifacts` only affects topology/node/edge CSV dumps.
- Dependencies/performance: document external deps (e.g., `mkdssp`), Bio.PDB warnings (`--pdb-warnings`), and cost drivers/tunable params for heavier modules.
- Add unit tests to lock in feature_dim, ordering, branch behavior, and validation before relying on new modules in pipelines. Use `QTOPO_TEST_USE_REAL_DEPS`/`QTOPO_SKIP_MODULE_REGISTRY`/`QTOPO_ALLOW_MODULE_OVERRIDE` flags as needed in tests.
- Jobs precedence: leave `jobs` in params/defaults; framework resolves CLI `--jobs` > config `default_jobs` > module `jobs/auto`.
- Schema summary: topology columns are auto-extracted during the edge stage (when topology CSVs exist) and node_feature_columns are mirrored into `schema_summary.json`; if modules add explicit columns, ensure these surface consistently.
- CLI/config integration: wire new toggles through parser → feature selection → logging; document defaults vs overrides so users understand side effects.

## Advanced k-partite topology presets (quick guide)
- Module: `topology/persistence_k_partite_advanced/v1`
- Presets:
  - minimal: base 140D.
  - lean: base + cross-only + per-primary (no bias/polar/typed/secondary).
  - heavy: bias + polar H-bond + typed contacts; no secondary strat.
  - heavy_stratified: heavy + secondary strat (chemotype default); polar/typed on.
  - rich: heavy_stratified + weighted filtration ON by default; power/landmark stay opt-in to avoid unexpected cost/approximation.
- Defaults: polar/typed ON in heavy/heavy_stratified/rich; weighted ON only in rich; power/landmark OFF unless explicitly enabled. Generated configs include inline comments reflecting these choices.
