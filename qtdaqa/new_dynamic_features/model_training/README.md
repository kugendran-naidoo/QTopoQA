# Dynamic Model Training Pipeline

This directory contains the orchestration layer for training dynamic TopoQA
models. The tooling is built around a sweep → fine-tune workflow and exposes a
unified Python CLI (`train_cli.py`) plus a set of bash wrappers that bundle
common sequences.

## Quick start

```bash
# from qtdaqa/new_dynamic_features/model_training
./run_full_pipeline.sh
```

By default the pipeline:

1. Runs a manifest-driven sweep (`train_cli batch`) to explore several configs.
2. Picks the strongest checkpoint (selection metric aware).
3. Launches Phase 1 fine-tuning.
4. Fans out three Phase 2 fine-tunes (seeds 101/555/888).

Outputs are written under `training_runs/`, with the most recent run symlinked
at `training_runs/latest`.

### Development shortcuts

- Use `SKIP_SWEEP=1 SKIP_FINE=1` to dry-run manifest selection without training.
- Prefer `--fast-dev-run` or `limit_*_batches` overrides for smoke tests.
- `SAFE_RAMDISK_TEST_MODE=1` (when using the RAM-disk wrapper) avoids creating
  a real disk while still exercising the pipeline.

## Training run artefacts

Each run directory contains:

| Path | Notes |
| --- | --- |
| `config/config.yaml` | Final config after CLI overrides. |
| `config/original_config.yaml` | Copy of the source YAML. |
| `config/applied_overrides.yaml` | Only present when overrides were supplied. |
| `run_metadata.json` | Command, environment snapshot, trainer/dataloader settings, progress metadata. |
| `training_console.log` | Captured stdout/stderr from the Lightning trainer. |
| `feature_metadata.json` | Resolved node/edge schemas and graph builder provenance. |
| `dataset_coverage.json` | Coverage totals; training aborts if thresholds are not met. |
| `model_checkpoints/` | Checkpoints plus `best`, `second_best`, `third_best` symlinks named with the ranking metric. |

Older runs are rolled into `training_runs/history/` automatically.

## Script overview

### `run_model_training.sh`

Wrapper for the unified CLI. It resolves the repo root, exports `PYTHONPATH`,
and chooses `python3`/`python` unless you override the interpreter via
`PYTHON=/path/to/python`.

```bash
# Examples
./run_model_training.sh run --config configs/sched_boost_seed222.yaml
PYTHON=../../../../venv_qtopo/bin/python ./run_model_training.sh batch --manifest manifests/run_core.yaml
```

Direct module execution remains available:

```bash
python -m qtdaqa.new_dynamic_features.model_training.train_cli run --config configs/sched_boost_seed222.yaml
```

### `run_full_pipeline.sh`

End-to-end sweep + fine-tune orchestrator.

```bash
./run_full_pipeline.sh                   # use manifests/run_core.yaml
./run_full_pipeline.sh --manifest manifests/run_extended.yaml
SKIP_SWEEP=1 SKIP_FINE=1 ./run_full_pipeline.sh
RESUME_FROM=/abs/path/model.chkpt ./run_full_pipeline.sh
GRAPH_DIR=/alt/graphs NUM_WORKERS_OVERRIDE=2 ./run_full_pipeline.sh
```

Key environment toggles:

- `SKIP_SWEEP=1` – reuse existing sweep results; skip Phase 0.
- `SKIP_FINE=1` – stop after reporting the winning checkpoint.
- `RESUME_FROM=/path/model.chkpt` – bypass sweep discovery and fine-tune a
  specific checkpoint (warned if `SKIP_SWEEP` is omitted).
- `GRAPH_DIR=/path/to/graph_data` – forwarded to every `train_cli` call as
  `paths.graph`.
- `NUM_WORKERS_OVERRIDE=<int>` – overrides `dataloader.num_workers`.
- `PIPELINE_MANIFEST=/path/file.yaml` (or `--manifest`) – change the sweep plan.

Logs for each invocation are timestamped:
`full_pipeline_<context>_<timestamp>.log`.

### `run_fine_tune_only.sh`

Replays the Phase 1 + Phase 2 schedule without re-running a sweep.

```bash
./run_fine_tune_only.sh                                 # auto-picks best run
./run_fine_tune_only.sh --checkpoint /path/model.chkpt \
  --run-dir training_runs/sched_boost_seed222_...       # manual resume
```

- By default the script scans `training_runs/` and `training_runs/history/`
  for the lowest validation loss and resumes its best checkpoint.
- `--checkpoint` requires `--run-dir` to label outputs; both paths are validated.
- Creates new runs `<base>_finetune_phase1` and `<base>_phase2_seed{seed}`.

### `train_cli.py`

Unified Python entry point hosting the underlying training commands.

```bash
./run_model_training.sh run --config configs/sched_boost_seed222.yaml \
  --run-name demo_run --fast-dev-run --override dataloader.num_workers=0

./run_model_training.sh batch --manifest manifests/run_core.yaml \
  --override shared.overrides.paths.graph=/alt/graph_data

./run_model_training.sh resume --run-dir training_runs/demo_run

./run_model_training.sh summarise --run-dir training_runs/demo_run
./run_model_training.sh leaderboard --limit 5

# Direct module form (requires repo root on PYTHONPATH)
python -m qtdaqa.new_dynamic_features.model_training.train_cli summarise --run-dir training_runs/demo_run
```

Key subcommands (all share `--help`):

| Command | Purpose | Highlights |
| --- | --- | --- |
| `run` | Launch one job from a YAML config. | `--run-name`, `--fast-dev-run`, `--resume-from`, `--override key=value`, `--notes`, `--trainer-arg`. |
| `batch` | Execute a manifest of jobs. | Supports shared overrides, per-job overrides, fast-dev, and `continue_on_error`. |
| `resume` | Continue a paused run. | Points at an existing `training_runs/<id>` directory. |
| `summarise` | Emit run metadata/metrics as JSON. | Includes checkpoint ranking, runtime estimates, coverage, symlink targets. |
| `leaderboard` | Rank runs by selection metric. | `--limit`, `--run-root`, `--reverse`. |

Behavioural notes:

- Environment determinism flags (`PYTHONHASHSEED`, deterministic Torch) are
  injected automatically.
- Coverage defaults to fail-fast: if any split falls below the configured
  threshold (`coverage.minimum_percent`, default 100%), the trainer aborts with
  guidance to regenerate missing graphs or relax the threshold.
- Overrides adopt dotted-key syntax (`paths.graph=/path`, `trainer.num_epochs=10`).
- Run metadata captures trainer/dataloader configuration to support later
  comparisons and `monitor_best_model` reporting.

### `monitor_best_model.py`

Summarises the best checkpoint from a run directory and (optionally) follows it
as training progresses.

```bash
python monitor_best_model.py --run-dir training_runs/latest
python monitor_best_model.py --run-id sched_boost_seed222_... --follow --interval 60
python monitor_best_model.py --run-dir training_runs/demo_run --metrics-limit 10
```

Output is JSON with:

- Repository-relative paths to checkpoints and run directories.
- Winning metric (selection metric if enabled, otherwise `val_loss`).
- Learning parameters (LR, batch size, epochs, accelerator, etc.).
- Recent metric history, runtime estimates, and warnings when training is still
  in progress or checkpoints are missing.

Use `--follow` to poll until interrupted; `--interval` controls the polling
frequency.

## Working with configs & manifests

- Configs live in `configs/`. Fine-tune stages use
  `sched_boost_finetune*.yaml`, while sweep jobs rely on the sched\_boost
  variants referenced by the manifests.
- Manifests under `manifests/` define job batches. `run_core.yaml` is a
  compact sweep, whereas `run_extended.yaml` widens learning-rate and
  batch-size coverage.
- Always review `coverage.*` settings when pointing at a new graph directory.
  Set `coverage.fail_on_missing=false` and `coverage.minimum_percent=0` only
  when you intentionally want to ignore gaps.

## Suggested verification steps

Reference checklist (adapt / extend as needed):

1. `bash -n ./run_full_pipeline.sh` – syntax check before a long run.
2. `SKIP_SWEEP=1 SKIP_FINE=1 ./run_full_pipeline.sh` – confirm manifest parsing
   and checkpoint discovery without training.
3. `python -m train_cli run --config configs/sched_boost_seed777.yaml \
     --fast-dev-run --run-name smoke_test` – exercise the training stack with
   minimal batches.
4. `python -m train_cli summarise --run-dir training_runs/smoke_test` – inspect
   metadata, coverage, symlinks.
5. `python monitor_best_model.py --run-dir training_runs/smoke_test` – verify
   reporting in both JSON and human-readable form (`jq` friendly).

## See also

- `ramdisk_run_pipeline.sh` – optional wrapper to benchmark RAM-disk graph
  staging (reads defaults from `configs/ramdisk.yaml`).
- `run_manifest_sweep.sh` – helper to execute just the manifest stage.
- `tools/` & `tests/` – assorted utilities for hygiene scans and smoke tests.
