# Dynamic Feature Pipeline Overview

This folder contains the “dynamic” TopoQA stack. It automates three stages:

1. **Graph Builder (`graph_builder/`)** – extracts interface/topology/node/edge features and writes PyG `.pt` graphs plus metadata.
2. **Model Training (`model_training/`)** – consumes the graphs and metadata to train models (sweep + fine-tune).
3. **Model Inference (`model_inference/`)** – loads the best checkpoint, regenerates graphs if needed, and scores evaluation decoys.

The sections below explain how the pieces fit together and which files carry the metadata buses that keep everything in sync.

---

## Graph Builder (feature extraction)

*Where:* `qtdaqa/new_dynamic_features/graph_builder`  
*Docs:* `graph_builder/README.md`

- **Purpose:** convert raw PDB structures into graphs (PyTorch Geometric `Data` objects) containing interface coordinates, topology features, node features, and edge attributes.  
- **Wrapper:** `./run_graph_builder.sh` (exposes `--dataset-dir`, `--work-dir`, `--graph-dir`, `--log-dir`, `--jobs`, `--feature-config`). All paths are mandatory so accidental defaults never creep in.
- **Feature configs:** users pick one module per stage (`interface`, `topology`, `node`, `edge`). The config schema is validated up front (missing sections, malformed element filters, etc., cause the program to stop immediately). `--create-feature-config` writes a minimal template with inline instructions; `--list-modules` shows every available module ID and description.
- **Outputs:**  
  - `graph_dir/*.pt` – graphs with metadata embedded (`data.metadata`).  
  - `graph_dir/graph_metadata.json` – canonical schema (edge dim, module IDs, node columns).  
  - `log_dir/graph_builder_summary.json` + stage logs – track which modules ran, how many successes/failures, wall-clock times.

These metadata files are the contract with training/inference. Keep them next to the graphs whenever you move them.

---

## Model Training (sweep + fine-tune)

*Where:* `qtdaqa/new_dynamic_features/model_training`  
*Docs:* `model_training/README.md`

- **Purpose:** automate the full TopoQA training pipeline: run manifest sweeps, pick the best run with the chosen metric, and fine-tune.  
- **Wrappers:**  
  - `./run_full_pipeline.sh` – orchestrates sweep → Phase 1 → Phase 2 seeds.  
  - `./run_model_training.sh` – wraps `train_cli.py` for custom invocations (`run`, `batch`, `resume`, `summarise`, `leaderboard`).  
  - `train_cli.py` – unified Python entry point that handles config overrides, metadata logging, deterministic env settings, and Lightning invocation.
  - `./run_model_training.sh leaderboard --show-alt-selection` – optional flag that prints, for each entry, which checkpoint would win if `selection_metric` (val loss − Spearman bonus) were the primary metric, so you can compare objectives without rerunning training.
- **Config highlights:**  
  - `paths.graph` (required) – point to the builder’s `graph_dir`; training automatically loads `graph_metadata.json` from here, so it knows the node/edge dims and column names.  
  - `selection.primary_metric` – controls how sweeps/fine-tunes rank runs (`val_loss` default, or `selection_metric` that blends val loss + Spearman corr). `run_full_pipeline.sh` reads this setting from each run’s metadata so you never have to edit the shell script when switching metrics.  
  - `cache.enable_graph_cache` + `graph_cache_size` – default-on in-process cache for `torch.load`ed graphs; keeps macOS CPU runs deterministic and faster.  
  - `coverage.*` – fail-fast coverage checks so missing graphs don’t go unnoticed.
- **Outputs:**  
  - `training_runs/<run_id>/feature_metadata.json` – copies the schema from `graph_metadata.json`.  
  - `training_runs/<run_id>/model_checkpoints/*.ckpt` – each checkpoint embeds the same `feature_metadata`, so inference can recover the schema from the checkpoint alone.  
  - `dataset_coverage.json`, logs (`training.log`, `training_console.log`), metrics CSVs, MLflow artifacts (if enabled).  

If you plan to run inference later, don’t delete these run directories—they hold the feature metadata, best-checkpoint symlinks, and coverage reports that inference needs.

---

## Model Inference (scoring evaluation decoys)

*Where:* `qtdaqa/new_dynamic_features/model_inference`  
*Docs:* `model_inference/README.md`

- **Purpose:** take evaluation PDBs (e.g., HAF2, BM55), regenerate graphs with the exact same feature modules used during training, and produce DockQ predictions plus detailed ranking summaries.  
- **Wrapper:** `./run_model_inference.sh --config config.yaml.<dataset>`  
- **Config highlights:**  
  - `paths.data_dir` – evaluation structures.  
  - `paths.work_dir` – scratch folder for regenerated graphs and metadata.  
  - `paths.results_dir` – root directory for per-dataset results (`<results_dir>/<dataset>/…`).  
  - `paths.checkpoint` – optional; if omitted, inference scans `paths.training_root` and auto-selects the best checkpoint using each run’s recorded `selection.primary_metric`.  
  - `builder.jobs` / `options.reuse_existing_graphs` – control how the helper interacts with the graph builder when graphs must be regenerated.  
  - `paths.training_root` – where to look for training runs (`../model_training/training_runs` by default).
- **Automatic metadata plumbing:**  
  - Each training checkpoint stores `feature_metadata`. Inference reads it, writes `<work_dir>/feature_metadata.json`, and generates `builder_features/features.from_metadata.yaml` on the fly.  
  - `ensure_graph_dir` compares any cached graphs in `work_dir` against this metadata. If they mismatch, it reruns the builder using the generated config so the new graphs match training exactly.  
  - This means you can switch to new feature configs, retrain, and inference will adapt automatically—no manual edits.
- **Outputs:**  
  - Prediction CSV (`<results_dir>/<dataset>/inference_results.csv`) plus per-target ranking-loss/hit-rate summaries under `<results_dir>/<dataset>/<target>/`.  
  - Work directory contains `feature_metadata.json`, regenerated graphs (if needed), logs, and the auto-generated builder config for reproducibility.

---

## Metadata Chain at a Glance

```
feature-config.yaml  --(graph builder)-->  graph_metadata.json
                                        \-> graph_builder_summary.json

graph_metadata.json  --(model training)--> feature_metadata.json + checkpoints (embed schema)

checkpoint.feature_metadata --(model inference)--> work_dir/feature_metadata.json
                                                \-> builder_features/features.from_metadata.yaml
```

As long as you keep those metadata files with the graphs and checkpoints, every stage stays perfectly in sync—no manual edits needed when you tweak the feature config.
