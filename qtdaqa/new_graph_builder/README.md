# New Graph Builder (multi-scale edges)

This module regenerates the TopoQA training graphs with the enhanced interface + edge features described in the 0.04836 study.  It mirrors the original `graph_builder2` workflow (interface → topology → node features → graph export) but introduces:

* Multi-scale contact bands with rich edge attributes (distance, inverse distance, direction vector, chain/residue encodings, contact histograms, contact counts).
* Config-driven thresholds (`work_dir/config.yaml`) so every run is reproducible.
* Automatic manifest + log emission under `--log-dir`, including hashes of the active parameters.
* Deterministic edge dumps per model (`work_dir/edge_features/*.edges.csv`) so you can inspect every attribute outside PyTorch.
* Smoke tests (`python -m pytest qtdaqa/new_graph_builder/tests`) that exercise the edge builder without touching the heavy datasets.

## CLI usage

```
graph_builder.py [-h] --dataset-dir DATASET_DIR --work-dir WORK_DIR \
                 --graph-dir GRAPH_DIR [--log-dir LOG_DIR] [--jobs JOBS]
```

* `DATASET_DIR`: root with target folders that hold `.pdb` (or `.cif`) decoys.
* `WORK_DIR`: empty folder for intermediates (interface/topology/node/edge CSVs).  Place your `config.yaml` here to override defaults.
* `GRAPH_DIR`: destination for `.pt` tensors.
* `LOG_DIR`: location for timestamped run logs (defaults to `./logs`).
* `JOBS`: number of worker processes for heavy stages (defaults to 4).

All arguments are required except `--log-dir` and `--jobs`.

## Step-by-step workflow

1. **Prepare workspace**
   ```
   mkdir -p runs/multiscale_work
   cp qtdaqa/new_graph_builder/config.example.yaml runs/multiscale_work/config.yaml  # optional edit
   ```
   Expected outcome: `runs/multiscale_work` exists and (optionally) contains the config you want to tweak.

2. **Launch the builder**
   ```
   python qtdaqa/new_graph_builder/graph_builder.py \
       --dataset-dir datasets/training/adjusted/Dockground_MAF2 \
       --work-dir runs/multiscale_work \
       --graph-dir qtdaqa/model_training/ARM_cpu_only/graph_data_multiscale \
       --log-dir runs/multiscale_logs \
       --jobs 8
   ```
   Expected outcome: streaming log output plus a new timestamped folder inside `runs/multiscale_logs/graph_builder_*`.  The work dir gains `interface/`, `topology/`, `node_features/`, and `edge_features/` sub-folders while `.pt` graphs appear under the chosen graph dir.

3. **Inspect logs + manifest**
   * `runs/multiscale_logs/graph_builder_*/graph_builder.log`: end-to-end trace with parameter echo.
   * `runs/multiscale_logs/graph_builder_*/run_manifest.json`: machine-readable summary (counts, config, edge feature dimension).
   * `runs/multiscale_work/edge_features/*.edges.csv`: human-readable edge tables for each decoy.

4. **Verify with the smoke test (optional but recommended after edits)**
   ```
   python -m pytest qtdaqa/new_graph_builder/tests -k edge_builder
   ```
   Expected outcome: `1 passed` confirming the multi-scale edge builder keeps producing sane tensors.

## Configuration knobs

`work_dir/config.yaml` (optional) can override any of these defaults:

```yaml
interface:
  cutoff: 14.0
  coordinate_decimals: 3
topology:
  neighbor_distance: 8.0
  filtration_cutoff: 8.0
  min_persistence: 0.01
  dedup_sort: false
node_drop_na: false
edge:
  bands:
    - {label: short,  min_distance: 0.0,  max_distance: 6.0}
    - {label: medium, min_distance: 6.0,  max_distance: 10.0}
    - {label: long,   min_distance: 10.0, max_distance: 14.0}
  histogram_bins: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
  contact_threshold: 5.0
  include_inverse_distance: true
  include_unit_vector: true
  unit_vector_epsilon: 1.0e-8
  normalize: none
```

Every run logs the resolved configuration so you always know which thresholds produced a given checkpoint.

## Expected deliverables per run

| Location | Contents |
| --- | --- |
| `work_dir/interface` | `.interface.txt` residues (rounded to configured decimals) |
| `work_dir/topology` | `.topology.csv` Gudhi-based features |
| `work_dir/node_features` | Node CSVs (DSSP + topology) |
| `work_dir/edge_features` | Multi-scale edge CSV dumps |
| `graph_dir` | PyTorch Geometric `.pt` graphs with enhanced edges |
| `log_dir/graph_builder_*` | Console log, per-stage logs, manifest, per-model PT logs |

If any stage fails, the manifest enumerates the offending decoys so you can re-run just those after fixing source data.

