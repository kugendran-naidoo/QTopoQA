General Questions
- PhD defence
what is the full workflow of the code?

Workflow Overview (Plain Language)

Start with a folder of protein structures (.pdb or .cif files).
For each protein, the script extracts three sets of features, storing them in a workspace:
Interface features capture which residues are close across different chains.
Topological features describe the 3D “shape” of the interface residues via persistent homology.
Node features assemble per-residue descriptors (DSSP, one-hot encodings, etc.).
Everything is written to disk so that downstream tools (like write_pt_file.py) can assemble a PyTorch Geometric graph object for machine learning.

Workflow (Detailed Mapping to Code)

Argument parsing & setup (graph_builder2.py:520-577).
Mandatory folders: input structures (--dataset-dir), scratch space (--work-dir), output graph folder (--graph-dir), log area (--log-dir).
The script sets deterministic options (e.g., PYTHONHASHSEED=0) so runs are reproducible (graph_builder2.py:94).
Environment validation (graph_builder2.py:549-620).
Checks permissions and empties work_dir/graph_dir.
Builds a map from residue names to PDB files (graph_builder2.py:560-618).

Interface extraction (called where the log says “Beginning interface residue extraction”; actual worker is process_pdb_file in lib/new_calculate_interface.py:39-94).
Uses the legacy topoqa/src/get_interface.py to find residues whose C-alpha atoms are within the default 10Å threshold of another chain.
Results are rounded and sorted deterministically (graph_builder2.py:116-158) and saved as <model>.interface.txt.

Topological features (graph_builder2.py:713-814).
For each interface file, persistent homology is computed via new_topological_features.py, producing <model>.topology.csv.
Logs record successes/failures per model.
Node features (graph_builder2.py:822-912).
node_fea_df.py (the same module used by inference) combines DSSP secondary-structure info, solvent accessibility, amino-acid one-hot encodings, and the persistent-homology columns from the previous step.
Output is <model>.csv under node_features/.

Packaging
graph_builder2.py stops after producing the CSVs; graph_dir remains empty (“graph-only builder”).
Graph assembly happens later via write_pt_file.py, which stages the three files and calls topoqa/src/graph.py:create_graph to make <model>.pt.

Simple Trace with a Concrete PDB
Imagine a toy structure with two chains, A and B, each with three residues.

Input: toy_complex.pdb with residues A1, A2, A3 and B1, B2, B3.
Interface step:
Finds that A2, A3 are within 8–10Å of B1, B2.
toy_complex.interface.txt might contain:
c<A>r<2>R<LYS>  10.53  14.22   9.47
c<A>r<3>R<ASP>   8.11  12.08  11.60
c<B>r<1>R<SER>   8.77  17.15  12.00
c<B>r<2>R<GLU>  10.55  16.93  14.40
Topological step:
Uses these four residues as anchor points to build persistence diagrams for different atom sets and neighbour distances.
Outputs columns like f0_C_sum, f1_CN_mean, etc. in toy_complex.topology.csv.

Node step (staged copy of interface/topology inside temporary folders):
Runs DSSP to compute secondary structure and solvent accessibility for each residue.
Aligns with the interface IDs and combines features into toy_complex.csv (127 rows became 4 rows in this toy example).
Columns include ID, rasa, phi, psi, SS8_*, AA_*, followed by persistent-homology features.

Graph assembly (when write_pt_file.py is invoked later):
Copies the three files into a temporary directory, calls create_graph, which:
Loads the node CSV (PyTorch tensor x of shape [num_nodes, num_features]).
Calculates pairwise edges: for every interface residue pair within cutoff, adds an entry to edge_index (shape [2, num_edges]).
Stores per-edge descriptors in edge_attr (num_edges × 11 in the default config).
Saves a torch_geometric.data.Data object (Data(x, edge_index, edge_attr)) to <model>.pt.
Later scripts load the graph with torch.load('...pt') to feed GNN models.

Why is the topology features dependent on the interface features and the node features dependent on the topology features? Explain the relationship and in general is this a common aspect of a graph used for GNN's? Create a summary and detail level explanation as if its been shown to a layman. Take a simple but useful example of a small pdb file as input and explain how to do this process using the 10A interface cutoff and 8A and 8A topology cutoffs etc.

Summary (Plain Language)

Think of a protein complex like a crowd of people.
Interface features act as a guest list: we only remember people who stand within 10 Å of someone from another group (another chain).
Topology features describe how those guests are arranged—who’s nearer, who forms triangles—so they must know exactly which guests were invited.
Node features add personal details (hair colour, height), and they also attach the topology stats to each guest.
Because each stage depends on who made the list first, the outputs match one-to-one. This layered dependency is typical when constructing graph inputs for GNNs: you first decide which nodes belong in the graph, then compute higher-level relationships, then package everything into x, edge_index, and edge_attr.
Detailed Explanation

Stage	What it Needs	Why It Depends on the Previous Stage	Output
Interface extraction (new_calculate_interface.py)	Raw 3D structure (.pdb/.cif), default cutoff 10 Å	It is the first step; identifies residues exposed to other chains	<model>.interface.txt: one residue per line, ordered and rounded (graph_builder2.py lines 110–158)
Topology features (graph_builder2.py lines 704–818)	Structure + interface file	Only interface residues matter; persistent homology (new_topological_features.py) needs exactly that list to stay in sync	<model>.topology.csv: persistent homology features aligned to the interface IDs
Node features (graph_builder2.py lines 822–912)	Structure + interface + topology directories	Builds on both: DSSP/one-hot data (structure) + interface order (IDs) + topological columns. Everything is merged on the interface IDs	<model>.csv: final per-residue feature table
When you later call write_pt_file.py, the three files are copied into a temp directory, topoqa/src/graph.py:create_graph reads them in that same order, and writes a PyTorch Geometric Data object (x, edge_index, edge_attr).

Simple Example (Toy PDB)

Assume a toy complex with chain A (residues 1–3) and chain B (residues 1–3).

Interface (10 Å cut-off)

Distances show A2 and A3 are within 10 Å of B1 and B2.
Output:
c<A>r<2>R<LYS> 10.53 14.22 9.47
c<A>r<3>R<ASP> 8.11 12.08 11.60
c<B>r<1>R<SER> 8.77 17.15 12.00
c<B>r<2>R<GLU> 10.55 16.93 14.40
Topology (default neighbour distance 8 Å, filtration 8 Å)

new_topological_features.py uses these four coordinates, slices them into element subsets (C, N, O, etc.), and computes persistence diagrams (birth/death of 0D/1D features under distance ≤ 8 Å).
Result: a CSV with columns like f0_C_sum, f1_CN_mean, etc., one row per residue.
Node features

node_fea_df.node_fea runs DSSP on the structure, aligns residue IDs from interface.txt, and merges the topological columns.

Example row (simplified):

ID	rasa	phi	psi	SS8_*	AA_*	f0_C_sum	...
c<A>r<2>R<LYS>	0.32	-60°	-45°	…	…	1.25	…
Graph packaging (write_pt_file.py)

create_graph loads toy_complex.csv, builds a tensor x of shape [4, feature_dim], a distance-based edge_index connecting A2/A3 to B1/B2, and edge_attr with smoothed distance/element statistics.
torch.save(Data(...), 'toy_complex.pt'). A later script calls torch.load to feed this graph to a GNN.
Is This Common in GNN Pipelines?
Yes. Many pipelines first define which residues/atoms/nodes belong to the graph (interface selection), derive relational or structural features (topology), and finally decorate each node with a rich feature vector (node_features). Graph neural networks expect consistent ordering across these stages—a change in node order without recomputing edges/nodes leads to mismatched tensors, so each step protecting the order is standard practice.

Is This Common in GNN Pipelines?
Yes. Many pipelines first define which residues/atoms/nodes belong to the graph (interface selection), derive relational or structural features (topology), and finally decorate each node with a rich feature vector (node_features). Graph neural networks expect consistent ordering across these stages—a change in node order without recomputing edges/nodes leads to mismatched tensors, so each step protecting the order is standard practice.

Suggested Prompt for a Visualisation

“Create a diagram of the interaction graph for toy_complex: place chain A residues (A2 LYS, A3 ASP) on the left and chain B residues (B1 SER, B2 GLU) on the right. Draw edges for pairs within 10 Å (A2–B1, A2–B2, A3–B1, A3–B2) and label each edge with its distance. Include a legend showing which features (DSSP, persistent homology) are attached to each node.”


Prompt for Visualizing the Resulting Graph
When asking ChatGPT (or another visual tool) to sketch the graph, provide coordinates/order:

“Please draw a simple bipartite graph showing residue nodes from chain A and chain B in `toy_complex`. Place A2 (LYS) and A3 (ASP) on the left, B1 (SER) and B2 (GLU) on the right. Connect nodes that are within 10 Å: A2–B1, A2–B2, A3–B1, A3–B2. Label each node with its residue name and chain.”  

For larger real datasets, include (1) list of residues with chain labels; (2) edges defined by interface proximity; (3) optional feature annotations (e.g., solvent accessibility). This lets the assistant generate an accurate GNN-style visualization.



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


	  PDB Insertion Codes and Impact on .pt files

	  Big Picture

Protein structures label each residue with a chain ID (A/B/…) and a residue number. When extra residues are inserted between numbered positions, PDB files reuse the number but add an insertion code such as 30A, 30B, etc. (node_fea_df.py:113-149). Those letters keep the biological order without renumbering the whole chain.
The inference script topoqa/src/get_interface.py:12-64 collects interface residues into a Python set and then sorts them by (chain, residue number, residue name), silently ignoring the insertion code. Sets preserve no order, so any residue pair like 30, 30A, 30B comes out in the order decided by CPython’s hash table at run time.
Graph construction (topoqa/src/graph.py:10-34) simply reads the interface file, merges it with node/topology features, and preserves whatever order it received. When the same structure is processed twice with different insertion-code tie-breakers, you get the same data but listed in a different row sequence. A GNN can still use it, but the .pt bytes don’t match and downstream diffs light up.
Why Behaviour Differs Now

The refreshed graph_builder2.py rewrites each interface file by rounding coordinates and then sorting deterministically on (chain, residue number, residue name, insertion code) (qtdaqa/graph_builder2/graph_builder2.py:99-157).
Whenever a chain really has multiple residues with the same number—e.g. c<L>r<14>R<ASP> and c<L>r<14>i<L>R<ASP>—the new file orders them alphabetically by insertion code, while the legacy inference run preserved the hash-based order it happened to generate. The node and edge tensors follow suit, so hash-order vs. deterministic-order files compare “different” even though the underlying graph is the same.
Concrete Example

legacy interface extract (order from Python set)
c<L>r<14>i<D>R<ARG>
c<L>r<14>R<ASP>
c<L>r<14>i<L>R<ASP>
c<L>r<14>i<H>R<GLU>

deterministic rewrite
c<L>r<14>R<ASP>
c<L>r<14>i<D>R<ARG>
c<L>r<14>i<H>R<GLU>
c<L>r<14>i<L>R<ASP>
Both mention the same four residues; they’re just listed in a different order.

What to Do

For faithful training/inference going forward
Use the deterministic ordering already in graph_builder2.py. Every rerun now yields identical .pt files, making regression testing and debugging much easier. A GNN treats permuted nodes the same, so model quality is unaffected as long as you keep one ordering per dataset.
For byte-for-byte legacy reproduction
You must restore the original tie-breaking: skip the deterministic sort and run the pipeline with the identical Python interpreter and PYTHONHASHSEED that produced the archived files. Otherwise, reusing the old interface text files verbatim is the only way to recreate those exact tensors.
Key Takeaway for Non-Experts
Insertion codes are the letters PDB files add when several residues share the same number. The historical inference script didn’t fully sort by those letters, so their order changed unpredictably between runs. The new pipeline fixes the ordering, which is good for reproducibility but means the tensors no longer match the legacy artifacts byte-for-byte. The data itself hasn’t changed—only the row order—and modern GNN training will work the same with the new files.

Comparing .pt files

compare_pt_files.py does not rely on a raw byte diff. It loads each pair with torch.load, walks the resulting objects, and performs field-by-field comparisons (tensors, lists, dicts, scalars). Tensors must have identical shapes and their values must match exactly, because the default tolerances (abs_tolerance=0, rel_tolerance=0) require every element to be equal. If you prefer a looser check you can rerun the script with non-zero tolerances, but even then it is comparing numerical content—not just file bytes.

What actually changes?
Only the order of entries in each tensor. The shapes stay identical, and the underlying numbers are the same once you line them up properly. The mismatch you see is a permutation of node rows and the corresponding edge indices.

Why does it happen?
For residues with insertion codes (e.g. PDB entry 1BTH has chain L residue 14 with variants 14, 14A, 14L, etc.), the historical inference run preserved whichever order the Python set returned. The refreshed graph_builder2.py sorts those residues deterministically (graph_builder2.py:99-158). That reorder is what propagates through to x, edge_index, and edge_attr.

Concrete illustration (1bth_1)
In output/ARM/old_method_inference/work/interface/1bth_1.txt the tied residues appear as:

c<L>r<14>i<D>R<ARG>
c<L>r<14>R<ASP>
c<L>r<14>i<L>R<ASP>
c<L>r<14>i<H>R<GLU>
With the deterministic rewrite (output/ARM/283_failed/work/interface/1bth_1.interface.txt) the same block becomes:

c<L>r<14>R<ASP>
c<L>r<14>i<D>R<ARG>
c<L>r<14>i<H>R<GLU>
c<L>r<14>i<L>R<ASP>
When you load the .pt files:

base = torch.load('.../283_failed/work/pt_files/1bth_1.pt')
old  = torch.load('.../old_method_inference/work/pt_file_data/1bth_1.pt')

base.x.shape == old.x.shape  # both (127, 172)
but the rows don’t match:

# After permuting the new tensor to the old order:
perm = [index of each old ID in the new ID list]
torch.allclose(base.x[perm], old.x)  # → True
edge_index and edge_attr exhibit the same behaviour: once you apply the same permutation to node indices (and adjust the edge endpoints accordingly) they line up exactly.

Takeaway
Insertion-code ordering shuffles the row ordering, not the data itself. The GNN sees the same graph, and retraining on the new artifacts is fine—as long as you stick with one ordering per dataset.

Figure out how to interpret the model training Table:


   | Name         | Type             | Params | Mode 
-----------------------------------------------------------
0  | criterion    | MeanSquaredError | 0      | train
1  | relu         | ReLU             | 0      | train
2  | sigmoid      | Sigmoid          | 0      | train
3  | edge_embed   | ModuleList       | 384    | train
4  | embed        | ModuleList       | 5.5 K  | train
5  | conv1        | ModuleList       | 50.2 K | train
6  | conv2        | ModuleList       | 50.2 K | train
7  | conv3        | ModuleList       | 50.2 K | train
8  | fc_edge      | Linear           | 528    | train
9  | protein_fc_1 | ModuleList       | 3.1 K  | train
10 | fc1          | Linear           | 4.2 K  | train
11 | out          | Linear           | 65     | train
-----------------------------------------------------------
164 K     Trainable params
0         Non-trainable params
164 K     Total params
0.658     Total estimated model params size (MB)
33        Modules in train mode
0         Modules in eval mode