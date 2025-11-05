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
   python -m train_cli batch --manifest manifests/run_core.yaml
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
- `--manifest PATH` (or `PIPELINE_MANIFEST=PATH`) selects an alternate sweep manifest. The default
  is `manifests/run_core.yaml`; `manifests/run_extended.yaml` widens the seed coverage.

All three flags can be combined; for example `SKIP_SWEEP=1 RESUME_FROM=/path/best.chkpt ./run_full_pipeline.sh`
launches only the fine-tune stages with a hand-picked checkpoint.

### RAM-disk wrapper

When you want to compare the pipeline with and without a RAM-backed graph directory,
use `ramdisk_run_pipeline.sh`. The wrapper reads defaults from `configs/ramdisk.yaml`
(`ramdisk.size`, `ramdisk.headroom`, `ramdisk.name`, and per-mode `num_workers`),
stages the configured graph tree onto a temporary volume, and runs the pipeline twice:

1. Baseline run against the original graph path (unless `--skip-baseline` is passed).
2. A RAM-disk run with `GRAPH_DIR` pointing at the mounted volume.

Command-line switches allow you to override the config on the fly:

```
./ramdisk_run_pipeline.sh \
  --graph-dir /Volumes/Data/qtdaqa/new_dynamic_features/graph_builder/output/.../graph_data \
  --baseline-num-workers 0 \
  --ram-num-workers 2 \
  -- --some-pipeline-flag
```

- `--graph-dir` changes the source directory copied onto the RAM disk.
- `--manifest` forwards a specific manifest to the pipeline (mirrors `PIPELINE_MANIFEST`).
- `--size`, `--headroom`, and `--name` override the provisioning parameters.
- `--baseline-num-workers` / `--ram-num-workers` temporarily set `NUM_WORKERS_OVERRIDE`.
- Pass `--skip-baseline` if you only care about the RAM run.

The script respects an existing `GRAPH_DIR` in the environment (useful when the pipeline
is already configured for a bespoke dataset) and records timing/metadata snapshots under
`training_runs/ramdisk_benchmarks/` for later comparison. Set `SAFE_RAMDISK_TEST_MODE=1`
to dry-run provisioning on systems where attaching disks is restricted—the wrapper will use
the source directory directly while still exercising the pipeline. It also forwards
`PIPELINE_MANIFEST` (or the `--manifest` override) so both runs evaluate the same sweep.

### Focused sweeps & follow-up fine-tunes

The default manifest (`manifests/run_core.yaml`) keeps the sweep nimble:

- `sched_boost_seed222.yaml` – baseline configuration.
- `sched_boost_lr035_seed222.yaml` – higher learning rate probe.
- `large_batch_sched_boost_seed222.yaml` – batch-size stress test.

When you need the wider seed sweep (222/777/1337) and batch/LR combinations, point the pipeline at
`manifests/run_extended.yaml`:

```bash
./run_full_pipeline.sh --manifest manifests/run_extended.yaml
```

You can achieve the same via `PIPELINE_MANIFEST=manifests/run_extended.yaml ./run_full_pipeline.sh` or by
invoking `python -m train_cli batch --manifest manifests/run_extended.yaml` directly.

After a sweep completes, use `./run_fine_tune_only.sh` to replay the Phase 1/Phase 2 schedule against a chosen
checkpoint. The script auto-detects the best run in `training_runs/` when no arguments are supplied:

```bash
./run_fine_tune_only.sh
# or resume from a specific checkpoint
./run_fine_tune_only.sh \
  --checkpoint training_runs/sched_boost_seed222_2025-11-03_11-03-07/model_checkpoints/checkpoint.sel-0.69302_val-0.05648_epoch090.chkpt \
  --run-dir sched_boost_seed222_2025-11-03_11-03-07
```

The helper now chains Phase 2 fine-tunes from the best Phase 1 checkpoint automatically, producing
`<base>_phase2_seed{101,555,888}` runs that match the pipeline workflow.

### Manifest bundles at a glance

`run_manifest_sweep.sh` is a thin wrapper around `python -m train_cli batch` with a few presets:

```bash
# default: manifests/run_core.yaml
./run_manifest_sweep.sh

# fast smoke-test manifest (fast_dev_run enabled in YAML)
./run_manifest_sweep.sh --bundle experiments

# broader seed sweep with LR/batch probes
./run_manifest_sweep.sh --bundle extended

# forward additional batch arguments after --
./run_manifest_sweep.sh --bundle core -- --limit 2
```

Use `--fast-dev` to inject `--override fast_dev_run=true` regardless of the manifest defaults, or
`--manifest path/to/custom.yaml` when you want an ad-hoc sweep. Pair the sweep with
`./run_fine_tune_only.sh` to replay the staged fine-tunes without rerunning Phase 0.

### GPU / MPS quick-start

Prototype configs are available under `configs/` for experimenting with hardware acceleration:

- `configs/sched_boost_seed222_cuda.yaml` – single-GPU (CUDA) settings, mixed precision enabled.
- `configs/sched_boost_seed222_mps.yaml` – Apple Silicon (MPS) settings for macOS 13+.

Usage examples:

```bash
# CUDA
python -m train_cli run --config configs/sched_boost_seed222_cuda.yaml --run-name seed222_cuda_test

# Apple MPS
PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m train_cli run --config configs/sched_boost_seed222_mps.yaml --run-name seed222_mps_test
```

Before running, confirm your environment:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("MPS available :", torch.backends.mps.is_available())
PY
```

- For CUDA, install the matching NVIDIA driver and a `torch` build with GPU wheels (see https://pytorch.org/get-started/locally/). Adjust `dataloader.num_workers`, `trainer.devices`, and `precision` in the config to match your hardware.
- For Apple Silicon, keep precision at 32-bit and export `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops transparently fall back to CPU.

These configs are baselines—you can copy them and tweak optimiser settings or batch sizes as your hardware allows.

## Training CLI

The legacy bash wrappers now delegate to the Python CLI.  Invoke the CLI
directly for clarity:

```bash
python -m train_cli run --config config.yaml
python -m train_cli run --config configs/sched_boost.yaml --override seed=777
python -m train_cli batch --manifest manifests/run_core.yaml
python -m train_cli resume --run-id training_run_2025-10-28_18-16-16
python -m train_cli summarise --run-dir training_runs/latest
python -m train_cli leaderboard --limit 5
```

Key commands:

- `run` – single training job, optional overrides via `--override key=value`.
- `batch` – execute multiple jobs described in a YAML manifest. See
  `manifests/run_core.yaml` (core sweep) and `manifests/run_extended.yaml` (expanded sweep) for templates.
- `resume` – continue a run from its best (or explicit) checkpoint.
- `summarise` – print JSON with the best validation score, checkpoint, and
  resolved schema for a run.
- `dataset_coverage.json` is generated in each run directory summarising any
  labels missing `.pt` graphs so gaps can be closed quickly.
- Checkpoints now expose `best.ckpt`, `second_best.ckpt`, and `third_best.ckpt`
  symlinks for convenient inspection. The underlying filenames retain the signed
  selection metric (for example `checkpoint.sel--0.69959_val-0.05479_epoch128.chkpt`),
  so the most negative score always lines up with `best.ckpt`.
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
