# Graph Builder 2 – Module Architecture Notes

This document captures how feature modules are structured, registered, and consumed by the builder. It’s a living reference to speed up future module development.

## Layout
- `modules/` – implementations grouped by kind:
  - `modules/interface` (e.g., `interface/polar_cutoff/v1`)
  - `modules/topology` (e.g., `topology/persistence_basic/v1`)
  - `modules/node` (e.g., `node/dssp_topo_merge/v1`)
  - `modules/edge` (e.g., `edge/legacy_band/v11`, hybrid PH+Lap aggregation variants); deprecated modules are kept under `modules/edge/deregistered` and are not auto-registered.
- `modules/base.py` – base classes per kind and helpers:
  - `InterfaceFeatureModule`, `TopologyFeatureModule`, `NodeFeatureModule`, `EdgeFeatureModule`
  - `build_metadata(...)` to define module metadata (id, summary, description, inputs, outputs, parameters, defaults)
  - `validate_params` methods per module to coerce/validate YAML params
  - `metadata()` / `describe()` inherited from the base
- `modules/registry.py` – registration:
  - `@register_feature_module` decorator registers a class with a `module_id`
  - `instantiate_module(module_id, **params)` to create instances
  - `get_module_class(module_id)` to inspect classes
- `lib/features_config.py`, `graph_builder2.py` – load/validate feature-config YAML, instantiate modules via registry, and run stages in fixed order: interface → topology → node → edge.

## Feature Config (YAML)
Each required stage has a block:
```
defaults:
  jobs: 16       # optional global worker override

interface:
  module: interface/polar_cutoff/v1
  alias: "friendly name"
  summary: "short"
  description: "longer"
  params:
    cutoff: 10.0
    coordinate_decimals: -1
    jobs: 16
topology:
  module: topology/persistence_basic/v1
  params: { ... }
node:
  module: node/dssp_topo_merge/v1
  params: { ... }
edge:
  module: edge/legacy_band/v11
  params: { ... }
```
CLI `--feature-config` points to this YAML; required stages must be present. `defaults.jobs` is optional.

## Module Self-Description
Each module defines `_metadata = build_metadata(...)` with:
- `module_id`, `module_kind`, `summary`, `description`
- `inputs`, `outputs`
- `parameters` (names/descriptions)
- `defaults`

Each module implements `validate_params` to coerce types/check ranges. `describe()` exposes metadata for listing/README generation. Registration is via `@register_feature_module` and `module_id`.

## Current Examples (by kind)
- Interface: `interface/polar_cutoff/v1` – polar interface finder; params: `cutoff`, `coordinate_decimals`, `jobs`. Interface files include coordinates and are deterministically sorted; PDB warnings can be toggled via CLI (`--pdb-warnings`). Cost grows with cutoff/residue count; jobs precedence: CLI --jobs > config default_jobs > module default.
- Topology: `topology/persistence_basic/v1` – per-residue persistent homology; params: `neighbor_distance`, `filtration_cutoff`, `min_persistence`, `dedup_sort`, `element_filters`, `jobs`. Can sort output; default element_filters tuples mirror the baseline (C/N/O combos).  
- Cost drivers: PH cost increases with larger `neighbor_distance`/`filtration_cutoff` and additional `element_filters`. `dedup_sort` improves determinism with minor overhead.  
- Topology: `topology/persistence_laplacian_hybrid/v1` – PH 140D + Laplacian spectral block (default 32D). Params extend persistence_basic with Lap options (`lap_graph_mode`, `lap_distance_cutoff`/`lap_k_neighbors`/`lap_max_neighbors`, `lap_edge_weight` gaussian/inverse/binary with sigma, `lap_eigs_count`, `lap_heat_times`, `lap_moment_orders`, `lap_normalize`). Cost drivers: PH element_filters + Lap eigs_count/weights/normalize and neighborhood size (cutoff/kNN/max_neighbors).  
- Topology: `topology/lightweight_MoL/v1` – PH 140D + 8D unweighted Lap moments (mu1-4, kappa2-4). Params: PH params plus `lap_k_neighbors`, `lap_max_neighbors`, `lap_size_threshold`, `lap_estimator` (exact/slq), `lap_slq_probes`, `lap_slq_steps` (compat), `lap_profile`. Cost drivers: PH filters/distances; Lap moment estimation depends on neighborhood size and estimator/SLQ probes. Default topo dim 148.  
- Topology: `topology/persistence_k_partite_advanced/v1` – k-partite PH with typed-contact block and richer presets: minimal (140D); lean (base+cross/per-primary); heavy (bias+polar H-bond+typed); heavy_stratified (adds secondary strat, chemotype default); rich (heavy_stratified + weighted filtration ON by default; power/landmark opt-in). Defaults: polar/typed ON in heavy/heavy_stratified/rich; weighted ON only in rich; power/landmark OFF unless enabled.
- Node: `node/dssp_topo_merge/v1` – merges DSSP with topology; params: `drop_na`, `jobs`. Depends on external `mkdssp`; node IDs are canonicalized/sorted; `drop_na` may remove rows. Default node_dim ≈ 32 DSSP + topology_dim (e.g., 172 with 140D PH); jobs precedence CLI --jobs > config default_jobs > module default.  
- Edge:
- Edge: `edge/legacy_band/v11` – 11D (distance + 10-bin histogram). Params: distance_min/max, scale_features, jobs. Cost: histogram per edge; scaling per graph. Dumps are resorted by edge_runner (src,dst,distance).
- Edge: `edge/edge_plus_bal_agg_topo/v1` – legacy 11D histogram + topo concat/mean/abs-diff (+ optional norms/cosine/minmax). Params: distance_min/max, scale_histogram, include_norms/cosine, include_minmax, variant lean/heavy, jobs. Feature_dim scales with topology_dim; dumps resorted by edge_runner. Cost: histogram + aggregation proportional to topo_dim; heavy adds min/max overhead; deterministic ordering (src,dst,distance).
- Edge: `edge/edge_plus_bal_agg_topo/v1` (cost/determinism): Aggregation cost scales with topology_dim; heavy adds min/max overhead. Deterministic ordering (src,dst,distance); edge dumps resorted by edge_runner.
- Edge: `edge/edge_plus_min_agg_topo/v1` – legacy 11D histogram + topo concat/abs-diff (+ optional norms/cosine/minmax). Params: distance_min/max, scale_histogram, include_norms/cosine, include_minmax, variant lean/heavy, jobs. Feature_dim scales with topology_dim (default topo_dim≈140); dumps resorted by edge_runner.
- Edge: `edge/edge_plus_pool_agg_topo/v1` – legacy 11D histogram + endpoint aggregation + pooled neighbor aggregation (k-nearest interface residues). Params: distance_min/max, scale_histogram (legacy block only), pool_k, include_norms/cosine, include_minmax, variant lean/heavy, jobs. Feature_dim scales with topology_dim; heavy adds min/max on endpoint+pooled blocks. Deterministic ordering (src,dst,distance); dumps resorted by edge_runner. Cost drivers: histogram + aggregation O(topo_dim) and pooled mean cost grows with pool_k and interface size.
- Edge: `edge/edge_plus_lightweight_MoL/v1` – legacy 11D histogram + 5D Laplacian moments (mu1-3, kappa2, kappa3) on the induced bipartite pair neighborhood (0–10 Å window). Params: distance_min/max, scale_histogram, lap_size_threshold, lap_estimator exact/slq, lap_slq_probes/steps, lap_max_neighbors, lap_profile, jobs. Deterministic ordering; dumps resorted by edge_runner. Cost drivers: histogram + Lap moments on pair neighborhood; scales with interface size and lap_max_neighbors. Default dim: 16.
  - edge_plus_min_agg_lap_hybrid: lean 530 dims (11 + 3×172 + norms+cosine); heavy 874 dims (+min/max).
  - edge_plus_bal_agg_lap_hybrid: lean 702 dims (11 + 4×172 + norms+cosine); heavy 1,046 dims (+min/max).
  - edge_plus_pool_agg_lap_hybrid: lean 1,393 dims (11 + endpoint agg + pooled agg); heavy 2,081 dims (+min/max in both blocks). Cost grows with topo_dim=172, pool_k, and min/max blocks; deterministic ordering; dumps resorted by edge_runner.
  - edge_plus_bal_agg_lap_hybrid (cost/determinism): Cost scales with topo_dim=172; heavy adds min/max overhead; deterministic ordering (src,dst,distance); dumps resorted by edge_runner.
  - `edge/legacy_band/v11` – 11D histogram; params: `distance_min/max`, `scale_features`, `jobs`. Deterministic edge ordering by (src_idx, dst_idx, distance); optional edge CSV dumps.
  - Deregistered (reference only; not auto-registered): `edge/multi_scale_v24`, `edge/neo_v24`, `edge/legacy_plus_topo_pair` (relocated to `modules/edge/deregistered`).
- Topology (new): `topology/standalone_MoL_replace_topology/v1` – PH-free Laplacian-only spectral topology; per-residue neighborhoods with unweighted/gaussian/inverse Laplacian, 24 eigs default (32 suggested richer), entropy, raw+centered moments, heat traces, Fiedler stats, Kirchhoff proxy; optional multi-scale radii (e.g., 6/8/10 Å) multiply the block. Defaults: unweighted, symmetric normalized, single-scale at neighbor_distance, size_threshold 80 for exact→SLQ.
- Performance/guards: defaults to size_threshold 80 for exact→SLQ; `lap_max_neighbors` caps neighborhood size; degenerate/empty neighborhoods zero-fill. Add `lap_profile` if you want timing breadcrumbs.

## Pipeline Contract
- Interface modules expose `extract_interfaces(...)`.
- Topology modules expose `generate_topology(...)`.
- Node modules expose `generate_nodes(...)`.
- Edge modules expose `build_edges(...)`; `lib/edge_runner.py` orchestrates .pt output and optional edge CSV dumps.
- Stages run in order: interface → topology → node → edge.

### Module Contracts (per kind)
- Interface: takes PDBs, produces interface residue files (deterministically sorted). Honors CLI `--pdb-warnings`.
- Topology: takes interface + PDB, produces per-residue `.topology.csv` (supports sorting via `dedup_sort`/artifact sorting).
- Node: takes interface + topology + PDB, produces node CSV with `ID` column; sorts/canonicalizes IDs; depends on external `mkdssp`.
- Edge: `build_edges` must return `EdgeBuildResult` with `edge_index` shape (E, 2), `edge_attr` shape (E, F), and a metadata dict (e.g., `feature_dim`, `edge_feature_variant`). Edge ordering is expected to be deterministic (src_idx, dst_idx, distance). Edge modules should honor `dump_path` for optional edge CSVs (edge_runner sorts dumps when enabled).

### Registration & IDs
- `module_id` convention: `kind/name/version` (e.g., `edge/legacy_band/v11`).
- Decorate with `@register_feature_module` and place under `modules/<kind>/` so `--list-modules` and `--create-feature-config` can discover it.

### Param Validation & YAML
- Implement `validate_params` to coerce types, enforce ranges, and keep params YAML-friendly (primitives, lists/tuples).
- Provide sensible defaults in `_metadata`; document them in `parameters`/`defaults`.

### Sorting & Determinism
- Interface files are always sorted; edge ordering sorted by (src_idx, dst_idx, distance); node/topology CSVs are sorted when artifact sorting is enabled (default on; disable with `--no-sort-artifacts`).
- New modules should preserve deterministic behavior and not reintroduce nondeterminism.
- Planned edge aggregation variants should inherit the existing deterministic edge ordering and rely on the canonical interface/node/topology sorting already enforced.
- All future modules must conform to the deterministic edge ordering and the canonical sorting of interface/node/topology artifacts to keep outputs reproducible.

### Edge Dumps
- Edge modules may write per-edge CSVs when `dump_path` is provided. `edge_runner` will sort the dump CSVs; modules should simply write the rows (include src/dst/distance when possible).

### Metadata Expectations
- Training/inference consume `graph_metadata.json`: include `edge_metadata.feature_dim`, `edge_module`, `edge_params`, `node_feature_columns`, `node_feature_dim`, and `module_registry` (populated via `graph_builder_summary.json`). Populate metadata sensibly (feature_dim, variant, etc.). Schema summaries mirror node dims/columns and auto-load topology columns when present.

### Dependencies
- DSSP requires `mkdssp` on PATH (node stage). Bio.PDB warnings can be toggled via CLI (`--pdb-warnings`). Document external binaries if a new module depends on them.

### Testing Pattern
- Tests use stubbed deps by default (`QTOPO_TEST_USE_REAL_DEPS=0`). Set `QTOPO_TEST_USE_REAL_DEPS=1` to exercise real deps. New modules should have minimal param-validation/unit tests.

### CLI/Listing
- `--list-modules` and `--create-feature-config` pull from module metadata. Ensure `summary/description/parameters/defaults` are filled so autogenerated configs and listings remain accurate. Listing formats: `text`, `markdown`, or `json` for scripting. Config templates/listings now emit variant-aware dim hints (`# dim: …`) when derivable from module aliases; dynamic passthrough aliases suppress the hint.

### Performance Notes
- If a module is heavier (e.g., neighbor expansion, per-edge PH), note the cost drivers and tunable params to dial cost up/down.

## Outputs & Metadata
- `graph_metadata.json` – canonical schema consumed by training/inference (edge/node schema, module registry, sample graph info).
- `schema_summary.json` – human-readable snapshot; now populated with `module_registry` (prefers co-located `graph_builder_summary.json`).
- `graph_builder_summary.json/log` – run summary (module selections, counts).

## Notes for Future Modules
- Follow the existing pattern: define `_metadata`, implement `validate_params`, register with `@register_feature_module`, provide the stage method (extract/generate/build), and document parameters/defaults.
- Keep params YAML-friendly (primitives/lists), and ensure `validate_params` handles coercion and range checks.
- Add descriptive `summary` and `description` for `--list-modules` and README/catalog generation.
- After registering a new module, sanity-check that it appears in `--list-modules` and in the `--create-feature-config --include-alternates` output, and add a minimal test covering metadata/feature_dim for regression.
- Add unit tests to lock deterministic ordering, feature_dim, and param validation; ensure new modules appear in listings/templates.
- Prefer vectorized implementations and deterministic tie-breakers (lexicographic) for neighbor selection; respect `sort_artifacts`.
- Provide sensible defaults, aliases, and clear param comments via `config_template`; document feature_dim in metadata and keep CSV outputs ID-first.
- Keep optional enhancements behind parameters and off by default to avoid unexpected regressions.
- Schema summary: topology columns are auto-extracted during the edge stage (when topology CSVs exist) and node_feature_columns are mirrored into `schema_summary.json`; if modules add explicit column naming (e.g., edge), ensure these can be surfaced consistently.
- Ablations: modules like `topology/persistence_null/v1` should preserve the expected schema/dimensions (e.g., 140D) and still emit `topology_columns.json` and correct metadata so downstream stages keep working.

 Quick metadata sanity checklist (for new modules)
 - After wiring a new module, run a 1–2 PDB smoke and verify:
   * `graph_metadata.json` has `topology_feature_dim`, `node_feature_dim`, `edge_feature_dim`, `_topology_schema.columns`, `_node_schema.columns`, and `module_registry` (topology module ID should also appear in `_topology_schema` when registry is available).
   * `schema_summary.json` mirrors topology columns and node_feature_columns.
   * CSV spot-check: interface/topology/node/edge files have expected column counts; .pt loads with matching dims.
 - Use a small script to assert the dims are non-null (see codex_new_module_handoff_guidance.md) before larger runs.

### Dimensionality guide (defaults)
- Topology: `topology/persistence_basic/v1` → 140 dims; `topology/persistence_laplacian_hybrid/v1` → 172 dims (140 PH + 32 Laplacian).
- Node: `node/dssp_topo_merge/v1` → 172 dims (32 DSSP/basic + 140 topology); `node/dssp_topo_merge_passthrough/v1` → 32 DSSP/basic + all topology columns (dynamic; e.g., 204 with hybrid topo). Sorting/determinism preserved; drop_na optional; jobs precedence: CLI --jobs > config default_jobs > module default.
- Cost: DSSP + merge; sorting/determinism preserved; drop_na optional.  
- Edge (assuming hybrid topology topo_dim=172; prepend 11D legacy histogram):
  - edge_plus_min_agg_lap_hybrid: lean 530 dims (11 + 3×172 + norms+cosine); heavy 874 dims (+min/max).
  - edge_plus_bal_agg_lap_hybrid: lean 702 dims (11 + 4×172 + norms+cosine); heavy 1,046 dims (+min/max).
  - edge_plus_pool_agg_lap_hybrid: lean 1,393 dims (11 + endpoint agg + pooled agg); heavy 2,081 dims (+min/max in both blocks).

## Upcoming lean Laplacian-moment modules (MoL)
- Topology/lightweight_MoL/v1 (planned): keep 140D PH, add +8D unweighted Laplacian moments on cross-chain bipartite neighborhoods (cutoff default 8 A). Moments: mu1-4 and kappa2-4 (no per-eigen outputs; heat trace deferred). Unweighted normalized Laplacian; exact eigs when node_count <= size_threshold (default 80), otherwise SLQ (probes=8, steps=32). Metadata/feature_dim captured; deterministic sorting preserved. Suggested config template strings: alias "140D PH + 8D unweighted Lap moments (mu1-4, kappa2-4) = Topology 148D (lean MoL)"; summary "Persistent homology plus lean unweighted Laplacian moments per interface residue (no per-eigen outputs)"; description outlining cutoff 8 A, unweighted normalized Laplacian, exact→SLQ policy, and no heat trace in v1; # dim: 148.
- Node passthrough: reuse `node/dssp_topo_merge_passthrough/v1` for arbitrary topology dims (no new alias planned). Cost: DSSP + merge + per-file scaling of topology columns.
- Edge/edge_plus_lightweight_MoL/v1 (planned): prepend legacy 11D histogram; add +5D unweighted Laplacian moments on induced bipartite neighborhood of the residue pair (reuse 0-10 A window). Moments: mu1-3, kappa2, kappa3. Same exact→SLQ policy with size_threshold/probes/steps params; deterministic ordering and optional CSV dumps. Suggested config template strings: alias "Legacy 11D Edge + 5D unweighted Lap moments (mu1-3, kappa2, kappa3) on pair neighborhood = Edge 16D (lean MoL)"; summary "Legacy histogram plus lean Laplacian moment context on the induced bipartite pair neighborhood (unweighted)"; description covering 0-10 A window, unweighted normalized Laplacian, exact→SLQ policy; # dim: 16.
- Estimators: CBNE omitted in v1; SLQ/exact only. Size threshold and SLQ probes/steps are configurable params. Keep unweighted adjacency for stability/consistency.
- Schema/metadata: ensure `feature_dim`, estimator params, and module ids surface in `graph_metadata.json` and `schema_summary.json` so training/inference auto-adapt; update feature-config templates to include defaults/comments for size thresholds and SLQ settings.

## New module checklist
- Register with clear `module_id`, alias, summary, description, defaults; keep params YAML-friendly with validation/coercion.
- Ensure determinism: sorted IDs, lexicographic neighbor/edge ordering; respect `sort_artifacts` where applicable (interface/edge ordering always deterministic).
- Populate metadata (feature_dim, params, variant) and ensure schema_summary/graph_metadata include new node/edge/topology dims/columns; topology columns are auto-extracted during edge stage and node_feature_columns mirrored.
- Config templates: include alias/summary/description/# dim hints and helpful `param_comments` so `--create-feature-config --include-alternates` stays readable; ensure `--list-modules` (text/markdown/json) looks correct.
- Edge dumps: honor `dump_path`; write src/dst/distance; let `edge_runner` sort dumps.
- Dependencies/performance: document external deps (e.g., mkdssp), Bio.PDB warnings (`--pdb-warnings`), and cost drivers/tunable params for heavy modules.
- Tests: cover param validation, feature_dim, deterministic ordering, branch behavior, CSV dumps (if any), metadata/listing/templates. Use `QTOPO_TEST_USE_REAL_DEPS`/`QTOPO_SKIP_MODULE_REGISTRY`/`QTOPO_ALLOW_MODULE_OVERRIDE` as needed.
- Jobs precedence: leave `jobs` in params/defaults; framework resolves CLI `--jobs` > config `default_jobs` > module `jobs/auto`.
- Determinism flag: `--no-sort-artifacts` only affects topology/node/edge CSV dumps; interface/edge ordering remains deterministic.
- Logging/failure handling: wrap IO/parsing in try/except, record failures in stage `failures`, and use existing logging (avoid noisy prints).
- CLI/config integration: wire new toggles through parser → feature selection → logging; document defaults vs overrides so users understand side effects.
- Reuse/composition: prefer wrapping/extending existing modules and updating metadata/dims so downstream tools recognize new shapes.

## Aggregated-topology edge modules
- Implemented: `edge_plus_min_agg_topo`, `edge_plus_bal_agg_topo`, `edge_plus_pool_agg_topo` (lean + heavy variants). Each prepends the legacy 11‑D histogram and keeps deterministic edge ordering; heavy variants add min/max blocks.
- Preserve deterministic interface/topology/node ordering; don’t reintroduce nondeterminism.
