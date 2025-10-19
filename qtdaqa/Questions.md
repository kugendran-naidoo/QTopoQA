General Questions
- PhD defence

Why train/predict with DockQ over CAPRI scores?
What is the weaknesses associated with CPU vs GPU resource training ?

Interface cutoff (10 Å) – This is the Cα–Cα distance threshold used when discovering which residues from different chains are in contact. In qtdaqa/graph_builder2/graph_builder2.py:40-54 the value is stored as INTERFACE_CUTOFF, and every call to process_pdb_file passes it through (graph_builder2.py:419-441). Inside new_calculate_interface.py:154-177, Biopython parses the PDB, computes squared distances between residues on different chains, and retains only those within 10 Å (new_calculate_interface.py:109-138). The resulting interface residues are written to *.interface.txt for each structure.

Topology neighbor distance (8 Å) – Once an interface residue is selected, the persistent-homology pipeline gathers all atoms within this radius before building Vietoris–Rips and Alpha complexes. The constant is named TOPOLOGY_NEIGHBOR_DISTANCE (graph_builder2.py:48-61) and becomes the neighbor_distance field of TopologicalConfig (graph_builder2.py:459-468). Downstream, compute_features_for_residue in new_topological_features.py uses that value when running a neighbor search around the residue’s reference atom (new_topological_features.py:389-412).

Filtration cutoff (8 Å) – During PH feature extraction, H0 bars exceeding this death time are trimmed out. graph_builder2.py sets TOPOLOGY_FILTRATION_CUTOFF (graph_builder2.py:48-61), which is passed into the same TopologicalConfig. Later, _compute_features_for_atoms applies the cutoff when building the persistence diagrams so only loops that die within 8 Å contribute to the zero-dimensional features (new_topological_features.py:220-284). This filters out long-lived bars that extend beyond the local environment defined by the PDB coordinates.

extract_old_topo_features.py still uses the original defaults (neighbor_distance = 6.0 Å, cutoff = 8.0 Å), whereas the production inference pipeline in k_mac_inference_pca_tsne4.py sets both the neighborhood radius and the persistence cutoff to 8 Å (see run_topology_features, where NEI_DIS = 8 and the same value is passed as Cut).

Parameter roles:

neighbor_distance: the radius around each reference residue within which atoms are collected before building the point cloud for persistent-homology calculations. A larger radius includes more atoms/geometry in the analysis.
cutoff: the persistence-death threshold applied to H0 bars. Bars longer than this are discarded, so the cutoff controls how long-lived connected components must be to contribute to the 0‑dimensional summary statistics.

- **Why does the Interface vector not have a header like Topology and Node vectors?**
- **Does this help or hinder model training or is it irrelevant?**
- **Answer:**
	- Interface outputs are intentionally headerless because every downstream consumer treats them as a simple, line‑based list of interface residues plus coordinates:
	- **Inference path:** k_mac_inference_pca_tsne4.py writes files via cal_interface/process_pdb_file. Later, _generate_topology_features and _stage_node_feature_inputs read each line, split on whitespace, and pull the first token as a residue ID. A header row would be mistaken for real data and either raise an error or leak an invalid ID.
	- **Training path:** the legacy node/graph builders (node_fea_df.py, graph.py) also call pd.read_csv(..., names=[...]) or string parsing with the assumption of no header.
	    
	  Topo and node features are true CSV datasets with named numeric columns; headers are necessary so pandas (and ultimately PyTorch) know which statistic each value represents. Those files are fed directly into model training/inference and having column names ensures consistent ordering.  
	    
	  Because interface files are only intermediates and never passed directly to the model, adding a header provides no benefit—and would actually break the existing parsers. The current, headerless format is therefore both intentional and required for the rest of the pipeline to function correctly.