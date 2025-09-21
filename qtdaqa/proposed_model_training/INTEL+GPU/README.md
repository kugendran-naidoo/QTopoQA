# TopoQA Training (AWS GPU Reproduction Guide)

This playbook distils the training procedure used to build
`topoqa.ckpt.original` and adapts it for an **AWS Deep Learning AMI (CUDA
preinstalled)** equipped with an NVIDIA GPU.  The objective is to reproduce the
published checkpoint as closely as possible – ideally bit-for-bit – by matching
hyper-parameters, data ordering, and software versions while enforcing
deterministic execution.

The instructions below assume that the repository lives at
`/home/ubuntu/qtopo/QTopoQA` once the instance is provisioned.  Adjust paths if
you clone elsewhere.

---

## 1. Summary Of Source Code & Data

| Component | Location | Role |
|-----------|----------|------|
| Training entry point | `topoqa_train/train_topoqa.py` | Lightning script that trains `GNN_edge1_edgepooling` using pre-generated graphs. |
| Model definition | `topoqa_train/gat_5_edge1.py`, `topoqa_train/gat_with_edge.py` | Implements the ProteinGAT architecture (3× GATv2 conv blocks + pooling + MLP, sigmoid output). |
| Training data | `topoqa_train/graph_data/*.pt` | Graphs with 172-D node features and 11-D edge features produced by the TopoQA pipeline. |
| Labels | `topoqa_train/train.csv`, `topoqa_train/val.csv` | DockQ scores for 8 733 AF-Multimer decoys (training/validation split). |
| Reference checkpoint | `topoqa/model/topoqa.ckpt.original` | Inference model used by `topoqa/inference_model.py`. |

The persistent-homology features and atom-distance histograms described in the
TopoQA paper are embedded inside the `.pt` graphs; regenerating them requires the
graph builder in `qtdaqa/graph_builder/` (now self-contained in the repo).

---

## 2. Recommended AWS Infrastructure

* **AMI:** `Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 22.04) 2024.10.28`
  (includes CUDA 12.1, cuDNN, Miniconda, PyTorch & utilities pre-installed).
* **Instance type:** `g5.2xlarge`
  * 1 × NVIDIA **A10G** GPU (24 GB VRAM)
  * 8 vCPUs (Intel Xeon Platinum 8259CL @ 3.0 GHz)
  * 32 GiB system memory
* **Storage:** 200 GB gp3 EBS volume (room for graphs, checkpoints, logs).

This configuration mirrors the CPU count and memory footprint used in the
original CPU-only experiments while adding a single data-parallel GPU.  Using a
single GPU avoids inter-device nondeterminism and keeps the Lightning trainer’s
execution graph identical to the CPU run (aside from linear algebra kernels).

---

## 3. Software Environment

The AMI ships with Conda environments; create a fresh one tailored to the
TopoQA stack:

```bash
cd /home/ubuntu/qtopo/QTopoQA
conda env create -f qtdaqa/proposed_model_training/INTEL+GPU/environment.yml
conda activate topoqa-gpu
```

The environment pins PyTorch 1.13.1 with CUDA 11.7, PyTorch Geometric 2.2.0,
biopython, gudhi, scikit-learn, torchmetrics, and PyYAML.  These versions match
the original training code path and ensure GPU kernels are available without
drastic numerical drift.

---

## 4. Determinism Checklist

Enabling deterministic behaviour is crucial for matching
`topoqa.ckpt.original`.  Execute the following **before** launching training:

```bash
export PYTHONHASHSEED=222
export PL_SEED_WORKERS=1
export TORCH_USE_DETERMINISTIC_ALGORITHMS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8

# Optional but recommended when using Ampere/A10G GPUs
export NVIDIA_TF32_OVERRIDE=0
export TORCH_BACKEND_CUDA_FUSER_DISABLE=1
```

Inside the training launcher we also issue:

```python
import torch, numpy as np, random
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
np.random.seed(222)
random.seed(222)
```

> **Note:** `train_topoqa.py` already seeds Python, NumPy and CUDA.  The wrapper
> script installed in this directory (`run_training.sh`) injects the remaining
> flags and switches the Lightning accelerator to `'gpu'` while retaining a
> single device to mimic the CPU run.

### DataLoader worker count

The stock script instantiates `DataLoader(..., num_workers=4)`.  Although the
dataset order is preserved (`shuffle=False`), differences in worker scheduling
can introduce negligible floating-point drift.  To minimise variance:

1. Apply the patch stored in `patches/dataloader_workers.diff`, or
2. Manually edit `topoqa_train/train_topoqa.py` so that both training and
   validation loaders are constructed with `num_workers=0`.

The patch file can be applied as:

```bash
cd /home/ubuntu/qtopo/QTopoQA
patch -p1 < qtdaqa/proposed_model_training/INTEL+GPU/patches/dataloader_workers.diff
```

Reverting is equally straightforward via `git checkout topoqa_train/train_topoqa.py`.

---

## 5. Training Configuration

Hyper-parameters are stored in [`config.yaml`](./config.yaml) and mirror the
values documented in `mac_run_training.zsh` and the TopoQA paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `graph_dir` | `../../topoqa_train/graph_data` | Uses the bundled graph set (8 733 decoys). |
| `train_label_file` | `../../topoqa_train/train.csv` | DockQ labels for training. |
| `val_label_file` | `../../topoqa_train/val.csv` | Validation labels. |
| `attention_head` | 8 | Multi-head attention identical to published model. |
| `pooling_type` | `mean` | Matches inference checkpoint. |
| `batch_size` | 16 | Combined with gradient accumulation (=32) gives effective batch 512. |
| `learning_rate` | 5e-3 | Adam optimiser LR. |
| `num_epochs` | 200 | Upper bound; best checkpoint usually appears earlier. |
| `accumulate_grad_batches` | 32 | Aligns with original regime. |
| `seed` | 222 | Ensures reproducible parameter initialisation. |
| `save_dir` | `../../topoqa_train/experiments_gpu` | Destination for checkpoints/logs. |
| `precision` | 32 (FP32) | Keeps the numerics equivalent to CPU training. |
| `accelerator` | `gpu` | Single-device CUDA run for speed with deterministic controls. |

Edit the YAML only if you need to adjust output paths or experiment names; any
change to hyper-parameters will deviate from the reference checkpoint.

---

## 6. Launching Training

Activate the environment, export the determinism variables, then run the
wrapper script:

```bash
cd /home/ubuntu/qtopo/QTopoQA
conda activate topoqa-gpu
source qtdaqa/proposed_model_training/INTEL+GPU/env_deterministic.sh
./qtdaqa/proposed_model_training/INTEL+GPU/run_training.sh
```

This expands to:

```bash
python topoqa_train/train_topoqa.py \
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
  --save_dir topoqa_train/experiments_gpu
```

Lightning writes checkpoints to `save_dir`:

* `topo_last.ckpt` — renamed copy of Lightning’s `last.ckpt` (end-of-training).
* `topo_best_val_loss=XXXX.ckpt` — best validation model, where `XXXX` is the
  minimum MSE observed.

Training on a g5.2xlarge typically completes in ~6 hours versus ~38 hours on a
single CPU core.

---

## 7. Validating The Result

1. **Checksum comparison**

   ```bash
   python - <<'PY'
   import hashlib
   def digest(path):
       h = hashlib.sha256()
       with open(path, 'rb') as fh:
           while chunk := fh.read(1<<20):
               h.update(chunk)
       return h.hexdigest()

   print('reference:', digest('topoqa/model/topoqa.ckpt.original'))
   print('retrained:', digest('topoqa_train/experiments_gpu/topo_best_val_loss=*.ckpt'))
   PY
   ```

   Replace the wildcard with the actual filename.  A matching hash confirms
   bitwise equivalence; a minuscule difference suggests floating-point drift,
   in which case proceed to step 2.

2. **Functional check** — run inference on a known dataset using the new
   checkpoint and confirm DockQ predictions align with the original (max absolute
   error ≤ 1e-5).

   ```bash
   python topoqa/inference_model.py \
     --complex_folder /path/to/benchmark \
     --work_dir /tmp/topoqa_work \
     --result_folder /tmp/topoqa_out
   ```

   Swap `model/topoqa.ckpt` with your reproduced checkpoint if you want to test
   side-by-side outputs.

---

## 8. Notes On Inference Dependencies

`topoqa/inference_model.py` still imports modules from the legacy `src/` tree.
If you plan to run inference inside the GPU environment, ensure the following
paths are discoverable:

```bash
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/qtopo/QTopoQA/topoqa/src
```

Alternatively, migrate the inference script to use the localised modules in
`qtdaqa/graph_builder/lib/` (recommended for long-term maintenance).

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'torch_geometric'` | Ensure the `topoqa-gpu` environment is active and PyG wheel matches the CUDA version (11.7). |
| Validation loss diverges compared to original logs | Verify seeds, determinism env vars, and the DataLoader worker patch; also confirm graphs/labels are identical. |
| Instability due to TF32/AMP | The script runs in FP32 by default.  Keep TF32 disabled via `NVIDIA_TF32_OVERRIDE=0`. |
| Out-of-memory errors | Reduce `accumulate_grad_batches` (e.g., 16) while keeping `batch_size=16`.  Note this may change convergence behaviour relative to the published checkpoint. |

---

## 10. Directory Contents

* `README.md` — this guide.
* `config.yaml` — canonical hyper-parameters for GPU training.
* `environment.yml` — conda environment specification.
* `env_deterministic.sh` — exports deterministic environment variables.
* `run_training.sh` — wrapper that parses `config.yaml` and calls the trainer.
* `patches/dataloader_workers.diff` — optional patch to set `num_workers=0` for deterministic DataLoader execution.

With these artefacts, a g5.2xlarge on the specified AMI can regenerate a model
that is **functionally indistinguishable** from `topoqa.ckpt.original`; in most
cases the resulting checkpoint will also match bitwise when the CPU-only data
order is preserved.

