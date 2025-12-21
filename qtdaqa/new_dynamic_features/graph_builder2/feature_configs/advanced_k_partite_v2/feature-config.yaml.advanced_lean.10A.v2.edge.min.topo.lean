# Auto-generated feature-config template
# Pick one module per required stage. Edit parameter values as needed.
# Run `./run_graph_builder2.sh --list-modules` for detailed descriptions.
defaults:
  jobs: 16  # Optional global worker override; remove if unused.

options:
  topology_round_decimals: -1  # Round topology numeric columns to N decimals (0-15); -1 or omit to disable (recommended 12 when enabling).

# REQUIRED stage: interface
interface:
  module: interface/polar_cutoff/v1
  alias: "residue ID + 3D coordinates (x,y,z) default cutoff 10A | TopoQA"
  summary: "Polar interface detection using distance cutoff."
  description: "Identifies interface residues across chains using the polar interface routine (new_calculate_interface.process_pdb_file) with configurable distance cutoff and coordinate rounding."
  params:
    cutoff: 10.0
    coordinate_decimals: -1  # skip rounding to keep raw coords
    jobs: 16

# OPTIONAL stage: topology
topology:
  module: topology/persistence_k_partite_advanced/v2  # alias: Advanced k-partite PH (v2 lean) ≈ topo_dim 596
  alias: "Advanced k-partite PH (v2 lean) topo_dim 596"
  summary: "Fixed-width k-partite persistent homology with slot padding + missingness channels."
  description: "Produces a fixed-width topology CSV per structure by padding missing chain/pair/typed/stratified slots up to configured caps and prefixing every feature column with a block identifier. Always appends slot-level missingness channels (present/size/reason) so padded zeros are not ambiguous."
  params:
    preset: lean  # minimal=base only; lean=base+per-primary+per-pair (fixed slots); heavy=lean+cross-bias+polar+typed; heavy_stratified=heavy+secondary stratification
    neighbor_distance: 8.0
    filtration_cutoff: 8.0
    min_persistence: 0.01
    element_filters:
      - [C]
      - [N]
      - [O]
      - [C, N]
      - [C, O]
      - [N, O]
      - [C, N, O]
    element_filters_stratified:
      - - C
        - N
        - O
    dedup_sort: false
    jobs: 16
    secondary_partition: none  # none|dssp|chemotype (heavy_stratified only)
    k_max: 2  # Number of primary slots (chains) in fixed schema; missing slots padded with reason_missing_slot
    secondary_k_max: 4  # Max secondary classes used (DSSP fixed 3-state H/E/C; chemotype fixed 4-class POS/NEG/POLAR/HYDRO); extra classes truncated deterministically
    enable_cross_bias: false
    intra_penalty_mode: default
    intra_penalty_lambda: null
    max_atoms: 25000  # Guardrail: if atom_count > max_atoms, heavy blocks are padded with reason_guardrail_skip
    max_block_seconds: 60.0  # Time cap per block; on timeout, slot is padded with reason_timeout
    enable_polar_block: false
    polar_hbond_weight: false
    polar_hbond_weight_factor: 0.5
    polar_hbond_energy_cutoff: -0.5
    polar_hbond_inter_only: false
    enable_typed_block: false
    dssp_timeout_seconds: 10.0
    dssp_slow_threshold: 2.0
    write_slot_maps: false  # If true, write work/topology_slot_maps/<pdb_id>.slot_map.json for debugging
    max_topology_dim_allowed: null  # Optional safety: warn/fail if expected worst-case topo_dim exceeds this limit
    fail_fast_topology_dim: false  # If true, raise error when expected worst-case topo_dim exceeds limit; otherwise warn only

# REQUIRED stage: node
node:
  module: node/dssp_topo_merge_passthrough/v1
  alias: "Passthrough (all dims from topology module) = Node Dynamic Dims"
  summary: "DSSP/basic features merged with all topology columns (passthrough)."
  description: "Runs DSSP/basic feature extraction and merges with the provided topology CSV using all non-ID topology columns (no hardcoded list). Topology columns are MinMax scaled per file; output columns ordered as ID, DSSP/basic 32D, then topology columns in file order. Suitable for hybrid topology (PH+Lap) or any dynamic topo layout."
  params:
    drop_na: true  # matches fea_df_clean = fea_df.dropna() in both inference_model.py and k_mac_inference_pca_tsne4.py
    jobs: 16

# REQUIRED stage: edge
edge:
  module: edge/edge_plus_min_agg_topo/v1
  alias: "Legacy 11D Edge + 423D {(norm + cosine) from 140D PH} = Edge 434D (Lean)"
  summary: "Legacy 11D histogram plus per-residue topology concat+abs-diff+cosine (lean)."
  description: "Reuses per-residue topology vectors to build edge features without rerunning persistence. For each cross-chain edge, concatenates (u_topo, v_topo, |u-v|) and appends cosine similarity and optional endpoint norms. Legacy 11D histogram (distance + 10-bin atom distance histogram) is prepended. Deterministic edge ordering is preserved (src_idx, dst_idx, distance)."
  # dim: 434
  params:
    distance_min: 0.0  # Minimum Cα distance (Å); must be < distance_max
    distance_max: 10.0  # Maximum Cα distance (Å) to include an edge (default 10.0 Å)
    scale_histogram: true  # MinMax scale only the legacy distance+histogram block (default on)
    include_norms: true  # Include L2 norms of endpoint topology vectors (default on)
    include_cosine: true  # Include cosine similarity between endpoint topology vectors (default on)
    include_minmax: false  # heavy variant only; adds per-dimension min/max blocks
    variant: lean  # lean (default) or heavy (adds min/max block)
    jobs: 16  # Optional worker override (CLI --jobs takes precedence)
