export DATASET=BM55-AF2
export WORK_DIR=edge_pairing_topoqa_10.5A 
time ./run_model_inference.sh \
--dataset-name ${DATASET} \
--config qtdaqa/new_dynamic_features/model_inference/config.yaml.${DATASET} \
--work-dir qtdaqa/new_dynamic_features/model_inference/output/${WORK_DIR}/work \
--results-dir qtdaqa/new_dynamic_features/model_inference/output/${WORK_DIR}/results \
--log-level INFO 2>&1 | tee ${DATASET}_${WORK_DIR}_$(date +%Y%m%d_%H%M%S).log

# Dynamic Inference Pipeline

This directory contains the inference stack for dynamic TopoQA models. It takes
evaluation decoys (e.g., HAF2, BM55, AF-Multimer sets), looks up the best
training run, regenerates graphs that match the training feature schema, and
produces DockQ predictions plus ranking summaries.

The design goal is “no manual wiring”: inference reads the metadata embedded in
training checkpoints (`feature_metadata`) and automatically generates the exact
`feature-config.yaml` needed to rebuild graphs. You only need to provide an
evaluation dataset and, optionally, a specific checkpoint.

---

## Quick Start

```bash
cd qtdaqa/new_dynamic_features/model_inference
./run_model_inference.sh --config config.yaml.HAF2 --dataset-name HAF2
```

`config.yaml.*` files live alongside the script; copy one and adjust paths as
needed. Every config follows the same structure:

```yaml
paths:
  data_dir: ./datasets/evaluation/HAF2/decoy
  work_dir: ./output/work
  results_dir: ./output/results
  dataset_name: null            # override per run or via --dataset-name
  label_file: ./datasets/evaluation/HAF2/label_info.csv   # optional
  checkpoint: null                                       # auto-select best run
  training_root: ../model_training/training_runs

builder:
  jobs: 12
  sort_artifacts: true        # optional; set to false to append --no-sort-artifacts when rebuilding graphs

options:
  reuse_existing_graphs: false
  check_schema: false         # optional: validate checkpoint/builder/graphs and exit
  force_node_dim: null        # optional emergency override; prefer using checkpoint/graph metadata

batch_size: 32
num_workers: 0
```

At runtime you must provide `--dataset-name` (or set `paths.dataset_name` in the
config). `--results-dir` now points to a directory root; inference automatically
creates a `<results_dir>/<dataset_name>/` subdirectory (and the same for
`work_dir`) so concurrent runs for different datasets never collide.

If you omit `paths.training_root`, the CLI defaults to
`../model_training/training_runs`. Before doing any heavy work, inference now
verifies that this directory exists (and thus contains the training runs you
expect). If you run inference from a different checkout, update
`paths.training_root` so the guardrail doesn’t trigger.

Flags you can override on the CLI:

```bash
./run_model_inference.sh \
  --config config.yaml.HAF2 \
  --dataset-name HAF2 \
  --log-level DEBUG \
  --dump-metadata
```

`--dump-metadata` prints the checkpoint’s `feature_metadata` and exits (useful
for debugging). Otherwise the command runs the full pipeline.

---

## How It Works

1. **Checkpoint discovery** – If `paths.checkpoint` is omitted, inference scans
   `paths.training_root` for runs with `run_metadata.json`, reads each run’s
   `selection.primary_metric`, and picks the best checkpoint according to that
   metric (the same logic `run_full_pipeline.sh` used during training). You can
   still pin a specific `.ckpt` by setting `paths.checkpoint` explicitly.

2. **Metadata export** – The chosen checkpoint embeds `feature_metadata`
   (copied from the builder’s `graph_metadata.json`). Inference writes this to
   `<work_dir>/feature_metadata.json` for reference and converts it into
   `builder_features/features.from_metadata.yaml`. This YAML mirrors the builder
   config used during training. Node/edge schemas (dims and columns) are carried
   through and enforced; `force_node_dim` is available as a last-resort override.

3. **Graph reuse or regeneration** – `builder_runner.ensure_graph_dir` compares
   any cached graphs under `<work_dir>/graph_data` with the expected schema:
   - If they match, it reuses them (fast path).
   - If not, it reruns the builder with the generated config so the new graphs
     match training exactly.
   - `--check-schema` (or `options.check_schema: true`) validates checkpoint → builder → cached graphs
     and exits without building.
   - `QTOPO_REUSE_ONLY=1` (used by the batch script below) will prefer reuse even if
     the metadata source path differs but the feature params are otherwise compatible.

4. **Inference run** – `inference_topoqa_cpu.py` loads graphs via a PyTorch
   Geometric `DataLoader`, feeds batches to the checkpointed model, and writes
   predictions plus various reports.

Throughout the process the wrapper writes detailed logs under
`<work_dir>/<dataset_name>` (`feature_metadata.json`, builder logs, inference logs,
and per-target reports).

---

## Outputs

- `<results_dir>/<dataset_name>/inference_results.csv` – CSV listing each decoy,
  predicted DockQ, and (if `label_file` was supplied) the true DockQ for reference.
- `<results_dir>/<dataset_name>/<target>/` – per-target ranking summaries:
  - `result.csv` (model vs. predicted DockQ) sorted descending.
  - `hit.rate_result.csv` (hit counts above DockQ 0.23/0.49/0.80).
  - `ranking_loss_result.csv` (m*, m^, ranking loss) plus TOP-10 listings.
- `<work_dir>/<dataset_name>/feature_metadata.json` – the schema recovered from the
  checkpoint (handy for auditing and reproducing builder runs). Includes `edge_schema`
  and `node_schema` with dims/columns.
- `<work_dir>/<dataset_name>/graph_load_profile.json` (if enabled via builder options) –
  profile of slowest graph loads.

Keep these alongside `builder_features/features.from_metadata.yaml` if you plan
to reuse the regenerated graphs later.

---

## Tips & Troubleshooting

- **Mismatch errors** – If you see “Graph metadata is incompatible…”, it means
  cached graphs don’t match the checkpoint’s schema (wrong edge module, feature
  dimension, etc.). Remove `<work_dir>/graph_data` or set
  `options.reuse_existing_graphs: false` to force regeneration.
- **Schema dry-run** – Add `--check-schema` (or set `options.check_schema: true`)
  to validate the checkpoint schema, local builder support, and any cached graphs,
  then exit without building or scoring.
- **Reuse-only mode** – Setting `QTOPO_REUSE_ONLY=1` keeps existing graphs even if
  the metadata source path differs (useful when the builder location changed but
  the features are otherwise compatible). Prefer normal reuse checks first.
- **Selecting a specific checkpoint** – set `paths.checkpoint` to an absolute
  `.ckpt` path. You should also set `paths.training_root` to the folder
  containing that run so logs resolve correctly. If you add manual
  `edge_schema`/`topology_schema` overrides, inference compares them against
  the checkpoint’s embedded metadata and fails fast if they disagree, which
  prevents long builder runs with mismatched configs.
- **Inspecting metadata without running inference** – use
  `./run_model_inference.sh --config ... --dump-metadata` to print the checkpoint
  schema and exit. For deeper inspection, `tools/inspect_schema.py --checkpoint <ckpt> [--graph-dir <graph_dir>]`
  prints node/edge dims and schemas.
- **Need to regenerate graphs manually?** – the builder helper writes the exact
  config it used under `<work_dir>/builder_features/`. You can rerun the builder
  yourself for debugging.

---

## Batch wrapper: `vscode_exec_edge_pool_lean_10A.sh`

Location: `qtdaqa/new_dynamic_features/model_inference/vscode_exec_edge_pool_lean_10A.sh`

- Auto-selects the top `TOP_K` checkpoints (default 5) from `model_training` via `train_cli.rank_runs`.
- Runs datasets `BM55-AF2`, `HAF2`, `ABAG-AF3` by default; adjust the `DATASETS` array as needed.
- Builds graphs once per dataset for the first checkpoint, then reuses them for the remaining checkpoints; sets `QTOPO_REUSE_ONLY=1`/`REUSE_ONLY=true` so reuse is forced when params match.
- Derives `WORK_DIR` from checkpoint metadata (graph_metadata source) unless you set it manually; optional timestamping via `APPEND_TIMESTAMP=true`.
- Exports `TOP_K`, `QTOPO_REUSE_ONLY`, and `REUSE_ONLY` so the Python layer honors reuse-only behavior; uses `--reuse-existing-graphs` everywhere and fails fast if reuse would trigger a rebuild.
- Supports schema validation (`--check-schema` semantics) implicitly by running reuse prechecks before Stage 2; stops if reuse eligibility fails to avoid clobbering graph_data.

Run via:
```bash
cd qtdaqa/new_dynamic_features/model_inference
bash ./vscode_exec_edge_pool_lean_10A.sh
```
Review `nohup.out`/logs under `output/<WORK_DIR>/` for per-checkpoint/dataset progress.

---

## Relationship to Training & Builder

- Training checkpoints embed `feature_metadata`. Inference reads it and never
  assumes hard-coded feature dimensions.
- Graph builder metadata (`graph_metadata.json`) is what training copied into
  checkpoints. As long as you keep training runs intact, inference can always
  recover the feature schema.
- Because the inference wrapper regenerates graphs when necessary, you’re free
  to tweak `feature-config.yaml`, rerun the builder + training, and inference
  will adapt automatically—no YAML edits required on the inference side.
