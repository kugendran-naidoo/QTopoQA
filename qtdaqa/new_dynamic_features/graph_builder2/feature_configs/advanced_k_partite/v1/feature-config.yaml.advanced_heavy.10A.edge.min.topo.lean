# Auto-generated feature-config template
# Pick one module per required stage. Edit parameter values as needed.
# Run `./run_graph_builder2.sh --list-modules` for detailed descriptions.
defaults:
  jobs: 8  # Optional global worker override; remove if unused.

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
    jobs: 8

# OPTIONAL stage: topology
topology:
  module: topology/persistence_k_partite_advanced/v1
  alias: "Advanced k-partite PH (heavy) 700D (bias+polar+typed)"
  summary: "Advanced k-partite persistent homology with optional typed-contact and weighted/polar blocks."
  description: "Extends the full k-partite module with optional typed-contact blocks and hooks for weighted filtrations, power/landmark modes. Primary partitions are chains; optional secondary stratification (chemotype/DSSP). Presets keep backward compatibility while allowing richer chemistry-aware variants."
  # dim: 700
  params:
    preset: heavy  # minimal=140D, lean=base+cross-pair+per-partition, heavy=~700D (bias+polar+typed, no secondary), heavy_stratified=~900D (adds secondary strat)
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
    jobs: 8
    secondary_partition: none  # none|dssp|chemotype (ignored by minimal/lean/heavy; used by heavy_stratified)
    k_max: 4  # Cap on number of primary partitions; blocks beyond cap are skipped (lean/heavy/stratified)
    secondary_k_max: 4  # Cap on number of secondary partitions (stratified only); ignored otherwise
    enable_cross_bias: true  # Used in heavy/heavy_stratified; ignored by lean/minimal even if true
    intra_penalty_mode: default  # default(=nd/2), light(=nd/4), heavy(=nd/1.34), none(=0), custom(use intra_penalty_lambda numeric); ignored by lean/minimal
    intra_penalty_lambda: null  # Used only when intra_penalty_mode=custom; otherwise derived from neighbor_distance; heavy/heavy_stratified only
    max_atoms: 25000  # Guardrail to skip heavy blocks; ignored by lean/minimal
    max_block_seconds: 60.0  # Per-block time cap (s) for lean/heavy/stratified; ignored by minimal
    enable_polar_block: true  # Used in heavy/heavy_stratified; ignored by lean/minimal
    polar_hbond_weight: true  # Toggle H-bond weighting for polar block (heavy/heavy_stratified); ignored by lean/minimal
    polar_hbond_weight_factor: 0.5  # Distance multiplier for H-bonded polar atoms (default=0.5); heavy/heavy_stratified
    polar_hbond_energy_cutoff: -0.5  # DSSP H-bond energy cutoff (kcal/mol) (default=-0.5); heavy/heavy_stratified
    polar_hbond_inter_only: false  # If true, only weight inter-chain H-bonded atoms; heavy/heavy_stratified
    enable_typed_block: true  # Enable typed-contact PH (hydrophobic/salt/H-bond); heavy/heavy_stratified default ON
    dssp_timeout_seconds: 10.0  # Per-PDB timeout for DSSP call (skip if exceeded).
    dssp_slow_threshold: 2.0  # Log a note if DSSP exceeds this time (seconds).

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
  module: edge/edge_plus_min_agg_topo/v1  # alias: Legacy 11D Edge + 423D {(norm + cosine) from 140D PH} = Edge 434D (Lean) | Legacy 11D Edge + 703D {(norm + cosine+ minmax) from 140D PH} = Edge 714D (Heavy)
  alias: "Legacy 11D Edge + 423D {(norm + cosine) from 140D PH} = Edge 434D (Lean) | Legacy 11D Edge + 703D {(norm + cosine + minmax) from 140D PH} = Edge 714D (Heavy)"
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
    jobs: 8  # Optional worker override (CLI --jobs takes precedence)
