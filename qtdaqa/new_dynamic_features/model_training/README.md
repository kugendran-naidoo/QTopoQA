# Dynamic Model Training

This copy of the training stack no longer hard-codes the edge feature schema.  
When a run starts, `model_train_topoqa_cpu.py` inspects the `.pt` graphs and
their `graph_metadata.json` sidecar to determine:

- the active edge module, aliases, and effective feature dimension;
- the node feature dimension (used to size the first embedding layer);
- the module registry details recorded by the graph builder (aliases, job
  overrides, summaries).

If a run is launched against a graph directory that does not contain
`graph_metadata.json`, the loader falls back to analysing the `.pt` files
directly.  The resolved schema is injected into the Lightning model and is
written to `<run_dir>/feature_metadata.json` so inference can reuse it.

Configuration files only need to override behaviour such as
`edge_schema.use_layer_norm`; fields like `edge_schema.dim` are computed at
runtime from the graph metadata.  Optional keys `metadata_path` and
`summary_path` can be supplied in a YAML config for non-standard layouts, but
they default to `graph_dir/graph_metadata.json` and an auto-discovered
`graph_builder_summary.json` respectively.
