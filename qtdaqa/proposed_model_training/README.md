# TopoQA Training Reproduction Plan

This package captures everything we learned while reverse-engineering the
`topoqa.ckpt.original` checkpoint and lays out a deterministic recipe for
retraining the model from scratch.  The goal is to re-create an equivalent
checkpoint (identical or numerically indistinguishable) using only the assets
already present in this repository.

The material below summarises code dependencies, data requirements, runtime
configuration, and the exact commands that were used in the original training
scripts (`topoqa_train/k_mac_train_topoqa.py` + `gat_5_edge1.py`).  All
instructions assume the working directory is the repository root
`/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA`.

## 1. Source Of Truth

| Component | Location | Notes |
|-----------|----------|-------|
| Graph builder that materialises training graphs | `qtdaqa/graph_builder/graph_builder.py` | Generates `.pt` graphs with 172-D node features and 11-D edge features. |
| Training script | `topoqa_train/k_mac_train_topoqa.py` | Lightning module that trains `GNN_edge1_edgepooling` with mean pooling and 8-head attention. |
| Model definition | `topoqa_train/gat_5_edge1.py` + `topoqa_train/gat_with_edge.py` | Three stacked GATv2 blocks with edge-aware attention; sigmoid regression head. |
| Reference checkpoint | `topoqa/model/topoqa.ckpt.original` | Saved Lightning checkpoint used by inference (`k_mac_inference_pca_tsne4.py`). |
| Training metadata | `topoqa_train/mac_run_training.zsh` | Documents canonical hyper-parameters used for the published model. |
| Training data manifests | `topoqa_train/train.csv`, `topoqa_train/val.csv` | Provide DockQ labels for 8,733 AF2/AF-Multimer decoys. |

## 2. Data Prerequisites

The published model was trained on graph data serialised under
`topoqa_train/graph_data`.  Each file corresponds to an interface graph built
from persistent-homology node features and atom-distance edge histograms as
described in the TopoQA manuscript.

To regenerate these graphs from raw PDBs you **must** use the bundled
graph-building pipeline *inside* `qtdaqa/graph_builder` (now self-contained in
`lib/`) with:

```bash
python qtdaqa/graph_builder/graph_builder.py \
  --dataset-dir <path_to_AF2_training_decoys> \
  --work-dir topoqa_train/graph_work \
  --out-graphs topoqa_train/graph_data \
  --log-dir topoqa_train/graph_logs \
  --use-local-json-config \
  --parallel 1
```

Because the original graphs were generated on CPU with a single worker, the
same settings should be preserved when reproducibility is required.  The JSON
configs under `qtdaqa/graph_builder/configs/` contain all the feature toggles.

## 3. Deterministic Runtime Environment

Training was executed on CPU only (`accelerator='cpu'`) with PyTorch Lightning.
To minimise nondeterminism:

1. Pin Python to 3.10 and use the `conda`/`mamba` environment defined in
   [`environment.yml`](./environment.yml).
2. Export the following before launching training:

   ```bash
   export PYTHONHASHSEED=222
   export CUBLAS_WORKSPACE_CONFIG=:16:8   # harmless on CPU; needed if you move to CUDA
   export PL_SEED_WORKERS=1
   export TORCH_USE_DETERMINISTIC_ALGORITHMS=1
   ```

3. Ensure the Torch backend stays on CPU to avoid cuDNN nondeterminism.  If a
   GPU environment is unavoidable, additionally set `CUDA_LAUNCH_BLOCKING=1`
   and `torch.backends.cudnn.deterministic = True` inside the training script.

## 4. Hyper-Parameters

The configuration distilled from `mac_run_training.zsh`, the paper, and the
Lightning module is encoded in [`config.yaml`](./config.yaml).  Key values:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `attention_head` | 8 | Matches published model and checkpoint metadata. |
| `pooling_type` | `mean` | Only pooling option used in released scripts. |
| `batch_size` | 16 | Same mini-batch size as `mac_run_training.zsh`. |
| `learning_rate` | 5e-3 | Adam optimiser LR from the original run. |
| `num_epochs` | 200 | Ensures convergence to the published checkpoint (best val loss observed well before 200). |
| `accumulate_grad_batches` | 32 | Effective batch size 512, as in the released script. |
| `seed` | 222 | Matches default in `k_mac_train_topoqa.py`; reproducibility hinges on not changing this. |
| `graph_dir` | `../../topoqa_train/graph_data` | Canonical training graphs. |
| `train_label_file` | `../../topoqa_train/train.csv` | DockQ labels. |
| `val_label_file` | `../../topoqa_train/val.csv` | Validation labels. |
| `save_dir` | `../../topoqa_train/experiments_repro` | Output folder (adjust as needed). |

## 5. Training Command

[`run_training.sh`](./run_training.sh) wraps the Lightning training script and
reads the hyper-parameters from `config.yaml`.  Example usage after activating
the conda environment:

```bash
cd /Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA
conda activate topoqa-training
./qtdaqa/proposed_model_training/run_training.sh
```

The script expands to the following Lightning command:

```bash
python topoqa_train/k_mac_train_topoqa.py \
  --graph_dir topoqa_train/graph_data \
  --train_label_file topoqa_train/train.csv \
  --val_label_file topoqa_train/val.csv \
  --attention_head 8 \
  --pooling_type mean \
  --batch_size 16 \
  --learning_rate 0.005 \
  --num_epochs 200 \
  --accumulate_grad_batches 32 \
  --seed 222 \
  --save_dir topoqa_train/experiments_repro
```

Training on CPU takes ~38 hours; wall-clock parity with the published checkpoint
is achieved by running on a single high-performance core (Lightning will use all
available threads but the results remain deterministic).

## 6. Validating The Checkpoint

After training, Lightning drops several checkpoints under `save_dir`:

* `topo_last.ckpt` — renamed Lightning "last" checkpoint.
* `topo_best_val_loss=<value>.ckpt` — copy of the best validation checkpoint.

To confirm equivalence with `topoqa.ckpt.original`:

```bash
python - <<'PY'
import hashlib, pathlib
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

orig = 'topoqa/model/topoqa.ckpt.original'
repro = 'topoqa_train/experiments_repro/topo_best_val_loss=0.00123.ckpt'
print('orig:', sha256(orig))
print('repro:', sha256(repro))
PY
```

Bitwise equality is expected when the same graph files, labels, seeds and
machine precision are used.  Minor floating point drift (e.g. on non-x86 CPUs)
may produce a near-identical checkpoint—verify by comparing validation loss and
the inference output from `k_mac_inference_pca_tsne4.py`.

## 7. Recommended Infrastructure

* **CPU:** Original training was executed on CPU (Lightning accelerator='cpu').
  For reproducibility use a deterministic BLAS implementation (MKL or OpenBLAS).
* **Memory:** ~14 GB for dataloaders + model states when using the provided
  graphs.
* **GPU (optional):** If you intend to accelerate with CUDA, enforce
  deterministic algorithms and consider replacing `torch_geometric` operators
  that are not deterministic; expect small numerical drift relative to the
  published CPU checkpoint.

## 8. Paper Alignment

The reproduced configuration matches the TopoQA manuscript:

* Node features: 172-D (32 basic + 140 PH statistics).
* Edge features: 11-D (10 atom-distance histograms + $C_\alpha$ distance).
* Model: three GATv2 layers with multi-head attention (heads=8), followed by
  mean pooling and an MLP regression head with sigmoid activation.
* Optimiser: Adam (lr=5e-3, weight decay=0.002 via module default).
* Loss: MSE vs DockQ scores.

## 9. Additional Advice

1. **Graph integrity:** The recorder features rely on `gudhi`, `Bio.PDB`, and
   `scikit-learn` MinMaxScaler.  Ensure those versions match the pinned
   environment to avoid numeric drift when regenerating graphs.
2. **WandB:** The training script contains commented `WandbLogger` code.  Keep
   it disabled unless you need cloud logging—re-enabling introduces additional
   nondeterministic factors.
3. **Checkpoints:** Lightning writes multiple files; archive the best checkpoint
   together with `config.yaml` and a manifest of the graph hashes for future
   verification.
4. **Inference smoke test:** After training, run `topoqa/k_mac_inference_pca_tsne4.py`
   against a known dataset and confirm identical DockQ predictions compared with
   the published checkpoint to ensure end-to-end equivalence.

---

For quick reference, the rest of this directory contains:

* [`config.yaml`](./config.yaml) — canonical hyper-parameters.
* [`environment.yml`](./environment.yml) — pinned package set for conda/mamba.
* [`run_training.sh`](./run_training.sh) — wrapper script that launches the
  Lightning trainer with the exact CLI options.

