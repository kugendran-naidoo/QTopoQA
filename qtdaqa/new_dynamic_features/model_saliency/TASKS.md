# Model Saliency Implementation Tasks

1. **Project bootstrap**
   - Set up Python package structure (module init, pyproject metadata if needed).
   - Capture shared constants (paths, defaults) in a dedicated config module.
2. **Metadata handling**
   - Load graph feature metadata (node/edge schemas, module groupings).
   - Resolve feature-group taxonomy from embedded module metadata or node CSV headers.
3. **Checkpoint and data access**
   - Load Lightning checkpoints and instantiate `CpuTopoQAModule` in eval mode.
   - Provide utilities to fetch graph `Data` objects and associated metadata/labels.
4. **Node & edge saliency (Captum)**
   - Implement Integrated Gradients / Gradient SHAP for node tensors.
   - Implement analogous routines for edge tensors with grouping aggregation.
   - Support per-residue, per-feature-group, per-contact reporting plus export helpers.
5. **Subgraph explanation & fidelity**
   - Integrate PyG `Explainer` (GNNExplainer) for contact/subgraph masks.
   - Compute deletion/insertion curves and randomization sanity checks.
6. **CLI / reporting**
   - Build command-line interface (e.g., `python -m model_saliency.run ...`) to drive analyses.
   - Emit structured outputs (JSON/CSV) and human-readable summaries under the run directory.
7. **Documentation & tests**
   - Document usage, configuration, and interpretation guidelines (README).
   - Add smoke tests or notebooks to verify saliency outputs on sample graphs.
