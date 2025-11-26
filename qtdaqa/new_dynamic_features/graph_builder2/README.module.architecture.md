# Graph Builder 2 – Module Architecture Notes

This document captures how feature modules are structured, registered, and consumed by the builder. It’s a living reference to speed up future module development.

## Layout
- `modules/` – implementations grouped by kind:
  - `modules/interface` (e.g., `interface/polar_cutoff/v1`)
  - `modules/topology` (e.g., `topology/persistence_basic/v1`)
  - `modules/node` (e.g., `node/dssp_topo_merge/v1`)
  - `modules/edge` (e.g., `edge/legacy_band/v11`, `edge/multi_scale/v24`, `edge/neo_v24`, `edge/legacy_plus_topo_pair`)
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
  jobs: 8        # optional global worker override

interface:
  module: interface/polar_cutoff/v1
  alias: "friendly name"
  summary: "short"
  description: "longer"
  params:
    cutoff: 10.0
    coordinate_decimals: -1
    jobs: 8
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
  - `edge/multi_scale/v24` – 24D geometric; params for bands, histogram bins, include_inverse_distance, unit_vector options, contact thresholds; writes optional edge dumps.
  - `edge/neo_v24` – hybrid 24D hist + multi-scale geometry; params include bands, histogram_bins (truncated), legacy_histogram_bins, histogram_mode (density | density_times_contact | density_and_count), contact_thresholds, include_inverse_distance, include_unit_vector, unit_vector_epsilon, scale_features, contact_normalizer, short_contact_max, long_band_mask, jobs.
  - `edge/legacy_plus_topo_pair` – legacy + per-edge pair topology (PH over residues + optional neighbors within `neighbor_distance`); params: `distance_min/max`, `scale_features`, `neighbor_distance`, `include_neighbors`, `filtration_cutoff`, `min_persistence`, `jobs`. Uses Gudhi for PH; can be slow when `include_neighbors` is true.

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
- `--list-modules` and `--create-feature-config` pull from module metadata. Ensure `summary/description/parameters/defaults` are filled so autogenerated configs and listings remain accurate.

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

## Planned (not implemented yet)
- Edge aggregation modules that reuse per-residue topology to build relational edge features without rerunning persistence:
  - Asymmetric: concat topo(u), topo(v)
  - Symmetric: mean, abs-diff, (optional min/max, cosine)
  - Optional pooling: mean over k nearest neighbors per endpoint, then mean/abs-diff (optional min/max/cosine) on pooled vectors
  - Augments legacy 11D histogram with deterministic, cheap relational topology
