# FLT Walkthrough (Graph Builder2 -> Model Training2 -> Model Inference)

This document provides a full, repeatable FLT (Feature Lottery Ticket) walkthrough
for your current pipeline. It assumes graph_builder2 now emits column labels for
topology, node, and edge features plus a validation manifest.

If you are new to this workflow, read it once end-to-end, then follow the steps
in order.

---

## 0) Identify the exact inputs (paths)

You need three inputs:

1) **Graph data directory** from graph_builder2 (contains `.pt` graphs and metadata).
2) **Baseline training run** directory from model_training2.
3) **Training config** used for the baseline run.

Example paths (replace with your own):

- Graph data:
  `/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/lap_lean_10A_ARM/graph_data`
- Baseline training run:
  `/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/model_training2/training_runs2/<baseline_run>`
- Training config:
  `/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/model_training2/configs/<config>.yaml`

If unsure about the baseline run, pick the most recent stable run with good
validation metrics.

---

## 1) Sanity-check graph metadata (required)

Purpose: FLT needs labeled feature columns. This step ensures they exist.

Run:

```bash
python qtdaqa/new_dynamic_features/graph_builder2/scripts/print_graph_builder_summary.py \
  /Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/lap_lean_10A_ARM
```

Confirm all of these are present and marked `[OK]`:

- `topology_columns.json`
- `node_columns.json`
- `edge_columns.json`
- `graph_manifest.json`
- `graph_metadata.json`

If any are missing, stop and fix the graph build before proceeding.

---

## 2) Audit for missing EMA metrics (recommended)

Purpose: If you align FLT baseline selection with inference Option‑B EMA metrics,
ensure EMA metrics exist so runs are not silently excluded.

Run:

```bash
python -m qtdaqa.new_dynamic_features.model_training2.tools.ema_metrics_audit \
  --training-root /path/to/training_runs2
```

Optional useful flags:
- `--metrics ema_val_loss ema_tuning_hit_rate_023` (explicit metrics)
- `--print-all` (print metric values per run)
- `--json /path/to/ema_audit.json` (save report)

If many runs are missing EMA metrics, prefer re‑running training with EMA enabled
or backfilling EMA metrics before proceeding.

---

## 3) Run FLT with your baseline run

Purpose: Rank feature groups and retrain pruned variants to measure impact.

Command template:

```bash
python qtdaqa/new_dynamic_features/model_training2/tools/feature_lottery_ticket.py \
  --config /path/to/your_training_config.yaml \
  --graph-dir /path/to/graph_data \
  --output-dir /path/to/flt_output \
  --baseline-run /path/to/your_baseline_run_dir \
  --keep-fracs 1.0,0.7,0.5,0.3 \
  --group-mode slot
```

Notes:
- `--group-mode slot` is recommended because your schemas are now labeled and
  fixed-width.
- `--keep-fracs` controls how aggressive pruning is. Smaller values mean fewer
  features kept.

What this does:
- Scores feature groups using the baseline run.
- Creates masks for each keep fraction.
- Retrains models using the masked features.

---

## 4) Review FLT outputs

Your `flt_output/` directory will contain:

- `flt_group_scores.json`
  - Ranked list of feature groups (most important first).
- `flt_summary.csv`
  - Metrics per keep fraction (loss, Spearman, etc).
- `flt_mask_keep_*.json`
  - The exact feature masks used for each keep fraction.
- Optional report files if enabled in the tool.

Start with `flt_summary.csv` and `flt_group_scores.json`.

---

## 5) Interpret results (simple, publishable rule)

Primary rule:

> Pick the smallest keep fraction that keeps validation loss within ~1% and
> Spearman within ~0.01 of baseline.

Secondary checks:
- **Stability across seeds**: top groups should remain near the top.
- **Consistency across splits**: training/validation/test should behave similarly.

If the smallest keep fraction is stable and close to baseline, those features
are likely dominant.

---

## 6) Decide add/remove features

Use the FLT ranking as your shortlist:

- **Remove** groups consistently ranked low with no measurable impact.
- **Keep** groups consistently ranked high and stabilizing metrics.
- **New blocks** should only stay if they improve metrics beyond noise.
- If a new block improves metrics but FLT drops it at small fractions, it may
  be redundant; consider disabling it.

---

## 7) Optional re-runs (more confidence)

Re-run FLT if you want:
- Different keep fractions (e.g., `0.9,0.8,0.6,0.4`)
- Multiple seeds (stability check)
- Different group modes (if your config supports it)

This can improve confidence but costs more compute.

---

## 8) Lock in a feature subset (final)

Once you decide a keep fraction, use the corresponding mask:

```yaml
feature_mask:
  path: /path/to/flt_output/flt_mask_keep_0.5.json
  apply_to: ["node", "edge", "topology"]
  strict: true
```

Then retrain and run inference as usual.

---

## Practical tips for novices

- **Do not skip Step 1.** Missing schema files cause FLT grouping to fail or
  produce meaningless results.
- Keep the **baseline run** fixed. It is the reference point.
- Only change **one thing at a time** (keep-fracs, seeds, group-mode).
- Keep `flt_output/` for record-keeping; it supports reproducibility.

---

## Quick checklist

- [ ] Graph metadata present (`topology_columns.json`, `node_columns.json`, `edge_columns.json`).
- [ ] `graph_manifest.json` present and validated.
- [ ] Baseline run path identified.
- [ ] FLT ran with keep-fracs.
- [ ] `flt_summary.csv` reviewed.
- [ ] Mask selected and applied for final training.
