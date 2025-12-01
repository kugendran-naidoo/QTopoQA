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
- Interface: `interface/polar_cutoff/v1` – polar interface finder; params: `cutoff`, `coordinate_decimals`, `jobs`. Interface files include coordinates and are deterministically sorted; PDB warnings can be toggled via CLI (`--pdb-warnings`).
- Topology: `topology/persistence_basic/v1` – per-residue persistent homology; params: `neighbor_distance`, `filtration_cutoff`, `min_persistence`, `dedup_sort`, `element_filters`, `jobs`. Can sort output; default element_filters tuples mirror the baseline (C/N/O combos).
- Node: `node/dssp_topo_merge/v1` – merges DSSP with topology; params: `drop_na`, `jobs`. Depends on external `mkdssp`; node IDs are canonicalized/sorted; `drop_na` may remove rows.
- Edge:
  - `edge/legacy_band/v11` – 11D histogram; params: `distance_min/max`, `scale_features`, `jobs`. Deterministic edge ordering by (src_idx, dst_idx, distance); optional edge CSV dumps.
  - Deregistered (reference only; not auto-registered): `edge/multi_scale_v24`, `edge/neo_v24`, `edge/legacy_plus_topo_pair` (relocated to `modules/edge/deregistered`).

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
- Training/inference consume `graph_metadata.json`: include `edge_metadata.feature_dim`, `edge_module`, `edge_params`, `node_feature_columns`, and `module_registry` (populated via `graph_builder_summary.json`). Populate metadata sensibly (feature_dim, variant, etc.).

### Dependencies
- DSSP requires `mkdssp` on PATH (node stage). Bio.PDB warnings can be toggled via CLI (`--pdb-warnings`). Document external binaries if a new module depends on them.

### Testing Pattern
- Tests use stubbed deps by default (`QTOPO_TEST_USE_REAL_DEPS=0`). Set `QTOPO_TEST_USE_REAL_DEPS=1` to exercise real deps. New modules should have minimal param-validation/unit tests.

### CLI/Listing
- `--list-modules` and `--create-feature-config` pull from module metadata. Ensure `summary/description/parameters/defaults` are filled so autogenerated configs and listings remain accurate. Listing formats: `text`, `markdown`, or `json` for scripting.

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

### Dimensionality guide (defaults)
- Topology: `topology/persistence_basic/v1` → 140 dims; `topology/persistence_laplacian_hybrid/v1` → 172 dims (140 PH + 32 Laplacian).
- Node: `node/dssp_topo_merge/v1` → 172 dims (32 DSSP/basic + 140 topology); `node/dssp_topo_merge_passthrough/v1` → 32 DSSP/basic + all topology columns (dynamic; e.g., 204 with hybrid topo).
- Edge (assuming hybrid topology topo_dim=172; prepend 11D legacy histogram):
  - edge_plus_min_agg_lap_hybrid: lean 530 dims (11 + 3×172 + norms+cosine); heavy 874 dims (+min/max).
  - edge_plus_bal_agg_lap_hybrid: lean 702 dims (11 + 4×172 + norms+cosine); heavy 1,046 dims (+min/max).
  - edge_plus_pool_agg_lap_hybrid: lean 1,393 dims (11 + endpoint agg + pooled agg); heavy 2,081 dims (+min/max in both blocks).

## Aggregated-topology edge modules
- Implemented: `edge_plus_min_agg_topo`, `edge_plus_bal_agg_topo`, `edge_plus_pool_agg_topo` (lean + heavy variants). Each prepends the legacy 11‑D histogram and keeps deterministic edge ordering; heavy variants add min/max blocks.
- Preserve deterministic interface/topology/node ordering; don’t reintroduce nondeterminism.
