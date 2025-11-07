# Dynamic Inference Pipeline

Inference consumes the `.pt` graphs produced by the dynamic graph builder and
the checkpoints emitted by the training pipeline. The current implementation
automatically regenerates compatible graphs when needed and can ingest per-run
metadata, making it adaptable to different feature sets without manual tweaks.

## Quick start

```bash
# from qtdaqa/new_dynamic_features/model_inference
./run_model_inference.sh --config config.yaml.HAF2
```

The CLI accepts either a YAML config (`--config config.yaml`) or direct CLI
arguments (`--data-dir`, `--work-dir`, `--checkpoint-path`, `--output-file`).
Configs follow this structure:

```yaml
paths:
  data_dir: /path/to/eval/structures
  work_dir: ./output/HAF2/work
  output_file: ./output/HAF2/results/inference_results.csv
  checkpoint: null        # optional, auto-select best training run if omitted
  training_root: ../model_training/training_runs  # optional override

builder:
  jobs: 12                # forwarded to graph builder

options:
  reuse_existing_graphs: false

batch_size: 32
num_workers: 0
```

### Metadata-aware builder

Each training checkpoint stores the feature metadata (module IDs, params, edge
schema) used for graph generation. When inference needs to regenerate graphs,
it now auto-creates a `features.from_metadata.yaml` in the work directory so
that the graph builder reruns with the *exact* same modules/parameters.

This means you can safely swap between 24‑D and 11‑D edges—or future feature
variants—without editing inference code. Keep the original graph metadata files
(`graph_metadata.json`, `graph_builder_summary.json`) so compatibility checks
remain fast when reusing cached graphs.

### Selection-aware checkpoint discovery

If `paths.checkpoint` is omitted, the CLI scans the configured training root and
picks the best checkpoint according to the metric recorded in each run
(`selection.primary_metric`). Runs trained with a custom selection metric are
automatically respected without editing inference configs.

## Output artifacts

Each run produces:

- A CSV of predictions (with optional ground-truth DockQ).
- Per-target ranking-loss and hit-rate summaries in `output/<target>/`.
- `feature_metadata.json` saved under the work directory for traceability.

## Testing

Minimal smoke tests live under `qtdaqa/new_dynamic_features/model_inference/tests`.
Use `pytest qtdaqa/new_dynamic_features/model_inference/tests` for <10 s
coverage of builder reuse, config parsing, and metadata guardrails.

## Testing

Minimal smoke tests live under `qtdaqa/new_dynamic_features/model_inference/tests`.
Use `pytest qtdaqa/new_dynamic_features/model_inference/tests` for <10 s
coverage of builder reuse, config parsing, and metadata guardrails.
