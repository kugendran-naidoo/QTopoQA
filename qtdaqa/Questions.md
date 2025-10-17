General Questions
- PhD defence

Why train/predict with DockQ over CAPRI scores?
What is the weaknesses associated with CPU vs GPU resource training ?

Interface cutoff (10 Å) – This is the Cα–Cα distance threshold used when discovering which residues from different chains are in contact. In qtdaqa/graph_builder2/graph_builder2.py:40-54 the value is stored as INTERFACE_CUTOFF, and every call to process_pdb_file passes it through (graph_builder2.py:419-441). Inside new_calculate_interface.py:154-177, Biopython parses the PDB, computes squared distances between residues on different chains, and retains only those within 10 Å (new_calculate_interface.py:109-138). The resulting interface residues are written to *.interface.txt for each structure.

Topology neighbor distance (8 Å) – Once an interface residue is selected, the persistent-homology pipeline gathers all atoms within this radius before building Vietoris–Rips and Alpha complexes. The constant is named TOPOLOGY_NEIGHBOR_DISTANCE (graph_builder2.py:48-61) and becomes the neighbor_distance field of TopologicalConfig (graph_builder2.py:459-468). Downstream, compute_features_for_residue in new_topological_features.py uses that value when running a neighbor search around the residue’s reference atom (new_topological_features.py:389-412).

Filtration cutoff (8 Å) – During PH feature extraction, H0 bars exceeding this death time are trimmed out. graph_builder2.py sets TOPOLOGY_FILTRATION_CUTOFF (graph_builder2.py:48-61), which is passed into the same TopologicalConfig. Later, _compute_features_for_atoms applies the cutoff when building the persistence diagrams so only loops that die within 8 Å contribute to the zero-dimensional features (new_topological_features.py:220-284). This filters out long-lived bars that extend beyond the local environment defined by the PDB coordinates.