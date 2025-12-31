# prevent temporary writes to /
export TMPDIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/temp
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

time GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder/output/24d_topoqa_settings.10A/graph_data \
TMPDIR="$TMPDIR" TEMP="$TEMP" TMP="$TMP" \
./run_full_pipeline.sh --manifest manifests/run_exhaustive.yaml | 
tee 24d_topoqa_10A_run_exhaustive_$(date +%Y%m%d_%H%M%S).log

# Dynamic Model Training Pipeline

This directory contains the orchestration layer for training dynamic TopoQA
models. It wraps PyTorch Lightning, handles sweep → selection → fine-tune, and
records all metadata needed for inference to stay in sync.

The tooling is built around three pieces:

- `train_cli.py` – unified Python CLI (commands: `run`, `batch`, `resume`,
  `summarise`, `leaderboard`). Handles config parsing, overrides, logging, and
  deterministic environment settings.
- `run_model_training.sh` – thin wrapper over `train_cli.py` for direct runs.
- `run_full_pipeline.sh` – higher-level orchestrator that runs a manifest sweep,
  picks the best checkpoint based on `selection.primary_metric`, launches Phase 1
  fine-tuning, then spawns Phase 2 runs for the configured seeds.

Outputs live under `training_runs2/`. Each run folder is self-contained: configs,
logs, metrics, coverage, checkpoints, and `feature_metadata.json`. Inference can
discover everything it needs (best checkpoint, feature schema) from these
directories.

---

## Quick Start

```bash
cd qtdaqa/new_dynamic_features/model_training2
GRAPH_DIR=/abs/path/to/graph_builder/output/<run>/graph_data \
./run_full_pipeline.sh
```

Environment overrides (prefix the command):

- `SKIP_SWEEP=1` – reuse existing sweep runs; jump straight to fine-tuning.
- `SKIP_FINE=1` – stop after reporting the winning checkpoint.
- `RESUME_FROM=/path/to/model.ckpt` – bypass sweep discovery and fine-tune the
  specified checkpoint (the script warns if `SKIP_SWEEP` isn’t set).
- `GRAPH_DIR=/path/to/graphs` – forwarded to every `train_cli` invocation as
  `paths.graph` (required; there is no default).
- `NUM_WORKERS_OVERRIDE=<int>` – overrides `dataloader.num_workers`.
- `PIPELINE_MANIFEST=/path/to/manifest.yaml` or `--manifest` – choose a different
  sweep manifest.

Syntax check before long runs: `bash -n run_full_pipeline.sh`.

---

## Config Guide (`configs/*.yaml`)

Key sections you’ll encounter:

### `paths`
- `graph` – builder output directory (must contain `graph_metadata.json`).
- `train_labels`, `val_labels` – CSVs with model IDs and DockQ scores.
- `save_dir` – where `training_runs2/<run_id>` will be created.

### `selection`
- `primary_metric` – choose how sweeps/fine-tunes rank runs:
  - `val_loss` (default) – best validation loss wins.
  - `selection_metric` – uses the Spearman-aware metric logged by Lightning
    (val loss adjusted by Spearman corr). Useful when correlation matters more
    than absolute loss.
  - `val_rank_spearman` / `val_rank_regret` – ranking metrics computed per-target
    on the validation split (requires target IDs in model names).
  - `tuning_rank_spearman` / `tuning_rank_regret` – ranking metrics computed on
    an explicit tuning slice (set `tuning.labels`).
  - `tuning_dockq_mae` / `tuning_hit_rate_023` – DockQ proxy metrics computed on
    the tuning slice; intended for Option‑B selection rather than primary training.
- `use_val_spearman` / `spearman_*` – control the secondary metric fed to the
  checkpoint callback.

### `model`
- `self_loops` – enable explicit self-loop edges in attention (default false).
- `self_loop_fill` – edge_attr fill policy for self loops (e.g., `mean` or `0.0`).
- `residual` – enable residual connections inside GATv2 layers.

### `tuning`
Optional, separate tuning split used for ranking-style selection without
touching the final eval sets.
- `labels` – CSV with the same columns as train/val (MODEL,dockq,capri).
- `eval_every` – compute tuning ranking metrics every N epochs.
- `max_samples` / `fraction` – cap the tuning slice size for faster metrics.
- `min_groups_ge2` – minimum number of multi-decoy target groups required
  before tuning metrics are allowed to drive selection (guardrail fallback).
Tuning metrics written each epoch (when enabled):
- `tuning_rank_spearman`, `tuning_rank_regret`
- `tuning_dockq_mae` (mean absolute error on DockQ)
- `tuning_hit_rate_023` (avg top‑10 hit‑rate for DockQ ≥ 0.23)

### `ranking_loss`
Optional pairwise ranking loss added to MSE during training.
- `weight` – 0 disables ranking loss (default).
- `margin` – hinge margin (only for `mode: hinge`).
- `mode` – `hinge` or `logistic`.
- `grouped` – only compare models within the same target ID.

### `variance_reduction`
Optional variance-reduced checkpointing after training.
- `enabled` – turn on top‑K averaging.
- `method` – `topk_avg`, `ema`, or `swa`.
- `top_k` – number of checkpoints to average.
- `ema_decay` – EMA decay factor (only for `method: ema`).
- `swa_start` – SWA start epoch or fraction of total epochs (only for `method: swa`).
- `save_every_epochs` – periodically persist EMA/SWA checkpoint to disk (0 = only at end).
- `resume` – resume EMA/SWA tracking from an existing averaged checkpoint if present.
- When EMA/SWA is enabled, a post-fit validation pass is run using the averaged checkpoint and
  the resulting metrics are stored in `run_metadata.json` under `ema_metrics` for eval-blind selection.
- To backfill `ema_metrics` for older runs, use:
  `python -m qtdaqa.new_dynamic_features.model_training2.tools.ema_backfill_eval --training-root training_runs2`
- Option B selection supports `--shortlist-metric` (default `best_val_loss`). Use `ema_val_loss` to build the shortlist
  based on EMA validation loss before applying the tuning metric.

### `dataloader`
- `num_workers` – default `0` for macOS (spawning workers can hurt stability).
- `batch_size`, `seed` – standard knobs.
- `cache` block:
  ```yaml
  cache:
    enable_graph_cache: true
    graph_cache_size: 256
  ```
  When enabled (default), `GraphRegressionDataset` keeps a bounded in-process
  cache of recently loaded graphs. This speeds up CPU-only runs while remaining
  deterministic (`num_workers=0`). Set `enable_graph_cache: false` to revert to
  legacy behavior.

### `trainer`, `optimizer`, `scheduler`, `early_stopping`
Standard Lightning knobs. Pay special attention to `num_epochs` and
`accumulate_grad_batches` to control total updates.

### `coverage`
Controls the fail-fast coverage checks. By default training aborts if any split
has missing graphs (`coverage.fail_on_missing: true`). You can relax this for
experimentation, but keep `dataset_coverage.json` around for auditing.

### `mlflow`
Optional logging; disable if you don’t need remote tracking.

---

## Workflow Details

1. **Sweep (`train_cli batch …`)** – executes the manifest jobs under
   `manifests/`. Each job gets its own `training_runs2/<run_id>` folder.

### Option B checkpoint selection helper
To select a checkpoint without using eval sets, use:
```
python tools/option_b_select.py --top-k 3 --tuning-metric best_val_tuning_rank_spearman
```
This picks top‑K by `val_loss` then chooses the best run by the tuning metric,
emitting the checkpoint path for that run.
You can also use DockQ proxies, for example:
```
python tools/option_b_select.py --top-k 3 --tuning-metric best_val_tuning_dockq_mae
python tools/option_b_select.py --top-k 3 --tuning-metric best_val_tuning_hit_rate_023
```
If EMA metrics are available, use `ema_tuning_dockq_mae` or `ema_tuning_hit_rate_023`.

### Recommended backfill order + comparisons (EMA metrics)
If you need to compare older EMA metrics against the new DockQ‑proxy EMA metrics:
1) Run the **legacy backfill** (no `--force`) and record your baseline inference results.
2) Run the **refresh backfill** (with `--force`) to overwrite `ema_metrics` and add:
   - `ema_tuning_dockq_mae`
   - `ema_tuning_hit_rate_023`
3) Re‑run Option‑B inference using the new metrics and compare summary CSVs:
   - `optionB_ema_sm_ema_val_loss_tm_ema_tuning_rank_spearman` (baseline)
   - `optionB_ema_sm_ema_val_loss_tm_ema_tuning_dockq_mae`
   - `optionB_ema_sm_ema_val_loss_tm_ema_tuning_hit_rate_023`

For **new runs**, no backfill is needed: the EMA post‑fit hook records
`ema_tuning_dockq_mae` and `ema_tuning_hit_rate_023` automatically.
Quick one‑liner to confirm after training completes:
```
python - <<'PY'
import json
from pathlib import Path
run = Path("training_runs2/<RUN_NAME>/run_metadata.json")
ema = json.loads(run.read_text()).get("ema_metrics", {})
print("ema_tuning_dockq_mae:", ema.get("tuning_dockq_mae"))
print("ema_tuning_hit_rate_023:", ema.get("tuning_hit_rate_023"))
PY
```
2. **Selection** – `run_full_pipeline.sh` inspects `run_metadata.json`/metrics
   and ranks runs using `selection.primary_metric`. The metric choice is recorded
   in each run’s metadata so inference knows whether val loss or the selection
   metric defined “best”.
3. **Phase 1 fine-tune** – restarts the winning checkpoint with the Phase 1
   config (default `configs/sched_boost_finetune.yaml`).
4. **Phase 2 seeds** – spawns multiple fine-tunes (default seeds 101/555/888).

Throughout the process, each run stores:

- `config/` – `config.yaml`, `original_config.yaml`, and `applied_overrides.yaml`.
- `feature_metadata.json` – copied from `graph_metadata.json` (includes edge_schema
  and node_schema with dims/columns); also embedded in every `.ckpt` under
  `checkpoint["feature_metadata"]` so inference can enforce matching node/edge dims.
- `model_checkpoints/` – `best.ckpt`, `second_best.ckpt`, `third_best.ckpt`
  symlinks plus full checkpoint files.
- `train_cli summarise/leaderboard` include a `builder_advisory` line (edge module,
  variant, builder id) so inference can confirm compatibility with the builder version in use.
- `dataset_coverage.json` – label vs. graph coverage for each split.
- Logs: `training.log` (file) and `training_console.log` (Lightning stdout).
- Metrics: CSV logs under `metrics/`, ready for plotting.

These artifacts let you pause/resume runs (`train_cli resume`), inspect progress
(`train_cli summarise`), or monitor training (`monitor_best_model.py`). They’re
also what inference uses to auto-select the best checkpoint later.

### Optional polish runs

When you want an even lower learning rate pass over the winning checkpoint,
use the new polish template:

1. Copy or edit `configs/polish_template.yaml`, set `dataloader.seed` to the
   seed/run you are polishing, and keep the slow-LR knobs (`lr=1e-3`,
   `num_epochs=360`, `scheduler.patience=4`, `early_stopping.patience=20`,
   `selection.primary_metric=val_loss`, Spearman secondary disabled).
2. Launch it via the single-job manifest:
   ```bash
   GRAPH_DIR=/abs/path/to/graph_data \
   SKIP_SWEEP=1 SKIP_FINE=1 \
   ./run_full_pipeline.sh --manifest manifests/run_polish_template.yaml
   ```
   (Feel free to override `PIPELINE_MANIFEST` instead of passing `--manifest`.)
3. After the run finishes, `train_cli leaderboard --output-format json --top 1`
   will include the polished checkpoint so you can feed it straight into
   inference.

---

## CLI Reference (Python layer)

```bash
./run_model_training.sh run --config configs/sched_boost_seed222.yaml \
  --run-name demo_run \
  --override cache.graph_cache_size=512 \
  --notes "quick smoke test"

./run_model_training.sh batch --manifest manifests/run_core.yaml \
  --override shared.overrides.paths.graph=/alt/graphs

./run_model_training.sh summarise --run-dir training_runs2/demo_run
./run_model_training.sh leaderboard --limit 5
# Show alternate selection-metric ranking alongside the primary metric view
./run_model_training.sh leaderboard --top 3 --show-alt-selection
# JSON output for scripting
./run_model_training.sh leaderboard --top 1 --output-format json > best_run.json
```

All commands ultimately call `python -m qtdaqa.new_dynamic_features.model_training2.train_cli ...`.

Useful options:
- `--override key=value` – dotted overrides (e.g., `paths.graph=...`).
- `--trainer-arg` – forward raw Lightning flags (`--trainer-arg log_every_n_steps=10`).
- `--fast-dev-run`, `--limit-train-batches`, `--limit-val-batches` – quick
  sanity checks.
- `--notes` – stored alongside `run_metadata.json` for documentation.
- `--show-alt-selection` (leaderboard only) – prints an extra line per entry
  showing which checkpoint would have ranked highest if `selection_metric`
  were the primary metric (useful for deciding whether to change the training
  objective before rerunning a sweep).

---

## Suggested Verification Steps

1. `bash -n ./run_full_pipeline.sh` – confirm the wrapper has no syntax errors.
2. `SKIP_SWEEP=1 SKIP_FINE=1 ./run_full_pipeline.sh` – dry-run manifest parsing
   and checkpoint discovery without launching Lightning.
3. `./run_model_training.sh run --config configs/sched_boost_seed777.yaml \
     --fast-dev-run --run-name smoke_test` – exercise a single job with minimal
   batches.
4. `./run_model_training.sh summarise --run-dir training_runs2/smoke_test` – view
   metadata JSON (best checkpoint, metrics history, coverage, etc.).
5. `python monitor_best_model.py --run-dir training_runs2/smoke_test --follow` –
   tail checkpoint progress in real time.
   The table mirrors the leaderboard layout: each update prints the primary
   metric line, the secondary metric status, raw `val_loss`/`selection_metric`,
   and (when applicable) an `alt_selection_rank` line showing the checkpoint
   that would win if `selection_metric` were primary.

Keep the resulting `training_runs2/` folders intact; inference relies on the
feature metadata and best-checkpoint links stored there.
