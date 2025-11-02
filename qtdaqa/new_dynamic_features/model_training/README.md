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

## Configuration layout

Configuration files are now organised into themed sections.  The default
`config.yaml` looks like this (abridged):

```yaml
paths:
  graph: ../../new_graph_builder/output/.../graph_data
  train_labels: ./train.csv
  val_labels: ./val.csv
  save_dir: ./training_runs

model:
  pooling_type: mean
  attention_head: 8

dataloader:
  batch_size: 16
  num_workers: 0
  seed: 222

trainer:
  accelerator: cpu
  devices: 1
  precision: 32
  num_epochs: 200
  accumulate_grad_batches: 16

optimizer:
  learning_rate: 0.005

scheduler:
  type: reduce_on_plateau
  factor: 0.4
  patience: 12

early_stopping:
  patience: 25

selection:
  use_val_spearman: true
  spearman_min_delta: 0.0
  spearman_weight: 1.0

logging:
  progress_bar_refresh_rate: 0
  log_every_n_steps: 100

mlflow:
  enabled: false
  tracking_uri: ./mlruns
  experiment: dynamic_topoqa
  run_name: null
  log_artifacts: true
  tags: {}
```

Legacy flat configs are still supported for backwards compatibility, but all
new configurations should follow the structured layout above.

### Secondary selection metric

By default, checkpoints are ranked using a composite score that favours low
`val_loss` while rewarding improvements in `val_spearman_corr`.  This behaviour
can be toggled under the `selection` block.  When enabled, the console prints a
clear notice showing the weight and minimum delta applied to the Spearman term.

### MLflow integration

The `mlflow` section governs optional experiment tracking.  When enabled and
`mlflow` is installed, the training loop logs hyperparameters, validation
metrics (including Pearson/Spearman correlations), and key artifacts (coverage
report, feature metadata, top checkpoints) to the configured tracking URI. If
the dependency is missing, a warning is emitted and training continues.

## Automated pipeline helper

The convenience script `run_full_pipeline.sh` orchestrates a full training cycle inside
`training_runs/` (and `training_runs/history/` for archived jobs). It was last validated
against the runs under `training_runs/`.

1. **Seed/config sweep** – kicks off every job described in the manifest and waits for
   them to finish. The fresh run directories land in `training_runs/`.

   ```bash
   python -m train_cli batch --manifest manifests/run_all.yaml
   ```

   After the sweep, the script inspects recent runs (or all runs when a sweep is skipped)
   and chooses the checkpoint with the lowest *selection metric* exposed by the trainer.

2. **Fine-tuning Phase 1** – resumes the best checkpoint with the phase-1 fine-tune
   config. The run is tagged `<best-run>_phase1`.

   ```bash
   python -m train_cli run \
     --config configs/sched_boost_finetune.yaml \
     --run-name "<best-run>_finetune_phase1" \
     --resume-from <path/to/best.chkpt>
   ```

3. **Fine-tuning Phase 2** – repeats the resume for each high-variance seed
   (`101`, `555`, `888`). Each launch reuses the same best checkpoint and stores the
   outputs as `<best-run>_phase2_seed<seed>`.

   ```bash
   for seed in 101 555 888; do
     python -m train_cli run \
       --config "configs/sched_boost_finetune_seed${seed}.yaml" \
       --run-name "<best-run>_finetune_phase1_seed${seed}" \
       --resume-from <path/to/best.chkpt>
   done
   ```

Run `./run_full_pipeline.sh` (from this directory) to execute all three stages sequentially.
Optional environment flags keep the behaviour flexible without editing the script:

- `SKIP_SWEEP=1` skips the manifest sweep and reuses the best checkpoint from existing runs.
- `SKIP_FINE=1` exits after reporting the winning checkpoint (useful when you only need the sweep).
- `RESUME_FROM=/abs/path/to/model.chkpt` forces the fine-tune phases to resume from a specific checkpoint.
- `GRAPH_DIR=/path/to/graphs` overrides the graph directory for every stage (the script forwards
  `--override paths.graph=...` to `train_cli`).

All three flags can be combined; for example `SKIP_SWEEP=1 RESUME_FROM=/path/best.chkpt ./run_full_pipeline.sh`
launches only the fine-tune stages with a hand-picked checkpoint.

## Training CLI

The legacy bash wrappers now delegate to the Python CLI.  Invoke the CLI
directly for clarity:

```bash
python -m train_cli run --config config.yaml
python -m train_cli run --config configs/sched_boost.yaml --override seed=777
python -m train_cli batch --manifest manifests/run_all.yaml
python -m train_cli resume --run-id training_run_2025-10-28_18-16-16
python -m train_cli summarise --run-dir training_runs/latest
python -m train_cli leaderboard --limit 5
```

Key commands:

- `run` – single training job, optional overrides via `--override key=value`.
- `batch` – execute multiple jobs described in a YAML manifest.  See
  `manifests/run_all.yaml` for an example.
- `resume` – continue a run from its best (or explicit) checkpoint.
- `summarise` – print JSON with the best validation score, checkpoint, and
  resolved schema for a run.
- `dataset_coverage.json` is generated in each run directory summarising any
  labels missing `.pt` graphs so gaps can be closed quickly.
- Checkpoints now expose `best.ckpt`, `second_best.ckpt`, and `third_best.ckpt`
  symlinks for convenient inspection.
- During training, the console emits a single mid-epoch progress line (25%,
  50%, 75%) plus a compact end-of-epoch summary showing train loss and the
  current best validation loss.

The wrapper scripts (`run_training.sh`, `run_all.sh`, etc.) now call the CLI
under the hood for compatibility with existing automation.

## Monitoring best checkpoints

Use `python monitor_best_model.py --run-id <run>` (optionally `--follow`) when
invoked from `qtdaqa/new_dynamic_features/model_training/`, or reference it via
`python qtdaqa/new_dynamic_features/model_training/monitor_best_model.py ...`
from the repo root to inspect the current best checkpoint, its filesystem
location, and the learning parameters that produced it.
