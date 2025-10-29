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

## Training CLI

The legacy bash wrappers now delegate to the Python CLI.  Invoke the CLI
directly for clarity:

```bash
python -m train_cli run --config config.yaml
python -m train_cli run --config configs/sched_boost.yaml --override seed=777
python -m train_cli batch --manifest manifests/run_all.yaml
python -m train_cli resume --run-id training_run_2025-10-28_18-16-16
python -m train_cli summarise --run-dir training_runs/latest
```

Key commands:

- `run` – single training job, optional overrides via `--override key=value`.
- `batch` – execute multiple jobs described in a YAML manifest.  See
  `manifests/run_all.yaml` for an example.
- `resume` – continue a run from its best (or explicit) checkpoint.
- `summarise` – print JSON with the best validation score, checkpoint, and
  resolved schema for a run.

The wrapper scripts (`run_training.sh`, `run_all.sh`, etc.) now call the CLI
under the hood for compatibility with existing automation.

## Monitoring best checkpoints

Use `python monitor_best_model.py --run-id <run>` (optionally `--follow`) when
invoked from `qtdaqa/new_dynamic_features/model_training/`, or reference it via
`python qtdaqa/new_dynamic_features/model_training/monitor_best_model.py ...`
from the repo root to inspect the current best checkpoint, its filesystem
location, and the learning parameters that produced it.

