#!/usr/bin/env python3
"""
Refactored FROM k_mac_inference_model.py, then includes dumping embeddings to run PCA and t-SNE
- use script 5_mac_batch_dump_embeddings_BM55-AF2.zsh to execute this
- unfortunately runs the entire pipeline just like any of the 4_mac_ranking_loss_hit_rate_* scripts
Fundamentally does not change all the TopoQA results that have been verified running k_mac_inference_model.py

- Wraps the whole pipeline in main()
- Exposes concurrency ("jobs") & DataLoader workers as CLI flags
- Adds robust device selection (CUDA / MPS / CPU)
- Normalizes I/O paths; validates inputs; better error messages
- Sorts & filters model lists deterministically
- Uses torch.no_grad(), model.eval(), and safe checkpoint loading
- Handles empty/failed graphs gracefully; writes results deterministically
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
import inspect

# pca, tsne patch
from typing import Dict
import csv
import torch.nn as nn
import re

# Prefer the modern DataLoader import path for PyG
try:
    from torch_geometric.loader import DataLoader  # PyG >= 2.0
except Exception:
    from torch_geometric.data import DataLoader  # fallback

from torch_geometric.data import Batch, Data

from torch.utils.data import Dataset

# Project-local imports
from src.get_interface import interface_batch
from src.topo_feature import topo_fea
from src.node_fea_df import node_fea
from src.graph import create_graph
from src.proteingat import GNN_edge1_edgepooling

import warnings  # NEW

# Silence Biopython DSSP noise when files are clearly PDB (we already pass file_type="PDB")
warnings.filterwarnings(
    "ignore",
    message=".*does not seem to be an mmCIF file",
    category=UserWarning,
    module="Bio.PDB.DSSP",
)

# Encourage loky to recycle workers to avoid rare memory/bad-state accumulation
os.environ.setdefault("LOKY_MAX_TASKS_PER_CHILD", "1")

# ---- [BEGIN PATCH: pca, tsne penultimate capture helpers] ----------------------------

class _PenultimateCapture:
    def __init__(self):
        self.z = None
    def hook(self, module, inputs, output):
        # For Linear: inputs is a tuple with one tensor [B, D]
        self.z = inputs[0].detach().cpu()

def find_final_linear_module(model: nn.Module) -> Optional[nn.Module]:
    # Heuristic: last nn.Linear with out_features == 1 or just the last nn.Linear
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    return last_linear

def attach_penultimate_hook(model: nn.Module):
    cap = _PenultimateCapture()
    final_lin = find_final_linear_module(model)
    if final_lin is None:
        raise RuntimeError("Could not find final nn.Linear in model to attach penultimate hook.")
    handle = final_lin.register_forward_hook(cap.hook)
    return cap, handle

# ---- [END PATCH: penultimate capture helpers] ------------------------------


# ---- [BEGIN PATCH: pca, tsne ] --------------------------------------------

# put this with the other small helpers
def normalize_decoy_id(s: str) -> str:
    """
    Make decoy/model identifiers comparable between filenames and labels CSV.
    - Drop file extension if present
    - Strip common curation suffixes appended to filenames (e.g., '_tidy')
    """
    s = str(s).strip()
    s = os.path.splitext(s)[0]  # drop .pdb/.cif/.mmcif if present
    # remove one trailing cleanup token if present
    s = re.sub(r'_(tidy|clean|cleaned|fixed|fix|renum|renumber|prepared|prep|relaxed|minimized|minimised|refined)$',
               '', s, flags=re.IGNORECASE)
    return s


def add_embedding_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        '--dump-embeddings',
        choices=['off', 'target', 'append-dataset'],
        default='off',
        help=(
            "off: no dump (default). "
            "target: write embeddings_{DATASET}_{TARGET}.npy (+ labels/dockq/ids). "
            "append-dataset: append to embeddings_{DATASET}.npy (+ labels/dockq/ids), creating if absent."
        )
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default=None,
        help='CSV with columns: decoy_id,dockq to label Acceptable+ (>=0.23) vs Incorrect.'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='./embeddings_out',
        help='Output directory for embeddings and labels.'
    )


# ---- [END PATCH: args additions] -------------------------------------------

# ---- [BEGIN PATCH: pca, tsne label utils] --------------------------------------------

def load_dockq_map(labels_csv: Optional[str], target_filter: Optional[str] = None) -> Dict[str, float]:
    """
    Load a mapping from decoy/model id -> DockQ.

    Accepts multiple schemas (case-insensitive):
      - decoy_id, dockq
      - model, DockQ [, target]
      - decoy, dockq
    If target_filter is provided and a target column exists, rows are filtered to that target.
    """
    m: Dict[str, float] = {}
    if not labels_csv:
        return m

    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return m

        # Preferred columns (lowercased keys)
        decoy_candidates = ["decoy_id", "model", "decoy", "model_id", "name"]
        dockq_candidates = ["dockq", "dock_q", "dockq_score"]
        target_candidates = ["target", "target_id", "pdb", "pdb_id"]

        for row in reader:
            row_l = { (k or "").lower(): (v or "") for k, v in row.items() }

            # Optional target filter
            if target_filter is not None:
                tcol = next((c for c in target_candidates if c in row_l), None)
                if tcol is not None and row_l[tcol].strip() != str(target_filter).strip():
                    continue

            dcol = next((c for c in decoy_candidates if c in row_l), None)
            qcol = next((c for c in dockq_candidates if c in row_l), None)
            if dcol is None or qcol is None:
                continue

            key = row_l[dcol].strip()
            
            key = normalize_decoy_id(key)     # <-- NEW

            try:
                val = float(row_l[qcol])
            except Exception:
                val = float("nan")

            if key:
                m[key] = val

    return m


def decoy_to_label(dockq: Optional[float]) -> int:
    if dockq is None or (isinstance(dockq, float) and (np.isnan(dockq))):
        return -1  # unknown for now
    return 1 if dockq >= 0.23 else 0
# ---- [END PATCH: label utils] ----------------------------------------------

# ---- [BEGIN PATCH: pca, tsne main integration] ---------------------------------------

def _append_or_write_npy(basepath: Path, fname: str, arr: np.ndarray) -> None:
    p = basepath / fname
    if p.exists() and p.stat().st_size > 0:
        old = np.load(p)
        arr = np.concatenate([old, arr], axis=0)
    np.save(p, arr)

def _append_or_write_txt(basepath: Path, fname: str, lines: list[str]) -> None:
    p = basepath / fname
    mode = 'a' if p.exists() and p.stat().st_size > 0 else 'w'
    with open(p, mode) as f:
        for s in lines:
            f.write(s + "\n")

def run_and_dump_embeddings(model, device, loader, model_order: list[str],
                            dataset_name: str, target_name: str,
                            labels_map: Dict[str, float], outdir: str,
                            mode: str):
    """
    mode: 'target' -> write per-target files
          'append-dataset' -> append to per-dataset files
    """
    os.makedirs(outdir, exist_ok=True)
    cap, handle = attach_penultimate_hook(model)
    model.eval()

    Z_chunks, Y_chunks, Q_chunks, IDS = [], [], [], []
    cursor = 0

    with torch.no_grad():
        for data in loader:
            # Ensure Batch and forward mirroring infer()
            if isinstance(data, (list, tuple)):
                # Your pipeline yields a single Batch, so this branch is unlikely here
                batch_list = [ _ensure_batch_vector(b.to(device)) for b in data ]
                out = model(batch_list)
                B = len(batch_list)
            else:
                b = _ensure_batch_vector(data.to(device))
                out = model([b])
                # How many graphs in this Batch?
                # PyG Batch stores 'ptr'; number of graphs = len(ptr) - 1, else infer as 1
                B = int(getattr(b, 'num_graphs', 1))

            # capture penultimate -> shape [B, D]
            z = cap.z
            if z is None:
                raise RuntimeError("Penultimate capture failed; check hook attachment.")
            Z_chunks.append(z.numpy())

            # decoy IDs for these B rows (since shuffle=False, this aligns with model_order)
            ids_b = model_order[cursor: cursor + z.shape[0]]
            IDS.extend(ids_b)
            cursor += z.shape[0]

            # Labels for these IDs (normalize both sides)
            ids_b_norm = [normalize_decoy_id(k) for k in ids_b]
            dq_map = labels_map                        # already normalized in load_dockq_map
            dq = [dq_map.get(k, float('nan')) for k in ids_b_norm]
            yb = [decoy_to_label(v) for v in dq]
            Q_chunks.append(np.asarray(dq, dtype=np.float32))
            Y_chunks.append(np.asarray(yb, dtype=np.int64))

    Z = np.vstack(Z_chunks) if Z_chunks else np.empty((0, 0), dtype=np.float32)
    Y = np.concatenate(Y_chunks) if Y_chunks else np.empty((0,), dtype=np.int64)
    Q = np.concatenate(Q_chunks) if Q_chunks else np.empty((0,), dtype=np.float32)

    # File naming
    if mode == 'target':
        tag = f"{dataset_name}_{target_name}"
        np.save(os.path.join(outdir, f"embeddings_{tag}.npy"), Z)
        np.save(os.path.join(outdir, f"labels_{tag}.npy"), Y)
        np.save(os.path.join(outdir, f"dockq_{tag}.npy"), Q)
        with open(os.path.join(outdir, f"decoy_ids_{tag}.txt"), 'w') as f:
            for k in IDS:
                f.write(f"{k}\n")
        print(f"[OK] Saved per-target embeddings as '*_{tag}.npy' in {outdir}")

    elif mode == 'append-dataset':
        # Append (or create) per-dataset cumulative files
        base = Path(outdir)
        _append_or_write_npy(base, f"embeddings_{dataset_name}.npy", Z)
        _append_or_write_npy(base, f"labels_{dataset_name}.npy",    Y)
        _append_or_write_npy(base, f"dockq_{dataset_name}.npy",     Q)
        _append_or_write_txt(base, f"decoy_ids_{dataset_name}.txt", IDS)
        print(f"[OK] Appended {len(IDS)} rows to '*_{dataset_name}.npy' in {outdir}")
    else:
        raise ValueError(f"Unknown dump mode: {mode}")

    handle.remove()

# ---- [END PATCH: main integration] -----------------------------------------




# ----------------------------- Utilities ----------------------------- #

def get_device(prefer: str = "auto") -> torch.device:
    """
    Choose the best available device.
    prefer: "auto" | "cuda" | "mps" | "cpu"
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] CUDA requested but not available. Falling back to MPS/CPU.")
    elif prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[warn] MPS requested but not available. Falling back to CUDA/CPU.")
    elif prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_models_from_dir(d: Path, suffixes: Sequence[str] = (".pdb", ".cif")) -> List[str]:
    """Return sorted model basenames (without extension)."""
    names = set()
    for s in suffixes:
        for f in d.glob(f"*{s}"):
            names.add(f.stem)
    return sorted(names)


# ---------------------- Pipeline wrappers (I/O) ---------------------- #

def run_interface_extraction(complex_dir: Path, interface_dir: Path, jobs: int) -> None:
    """Compute interfaces for all complexes in complex_dir into interface_dir."""
    interface_batch(str(complex_dir), str(interface_dir), int(jobs))

def run_topology_features(model_names: Sequence[str], complex_dir: Path, topo_dir: Path,
                          interface_dir: Path, jobs: int) -> None:
    """Compute persistent-homology (topo) features in parallel and write node_topo/<name>.csv."""
    topo_dir.mkdir(parents=True, exist_ok=True)

    # same element sets as in File I
    E_SET = [['C'], ['N'], ['O'], ['C','N'], ['C','O'], ['N','O'], ['C','N','O']]
    NEI_DIS = 8  # keep identical to File I's default

    def _one(name: str) -> None:
        try:
            pdb_path = _resolve_structure_path(name, complex_dir)
            if pdb_path is None:
                print(f"[topo_fea] skip {name}: no PDB/mmCIF in {complex_dir}")
                return

            vfile = interface_dir / f"{name}.txt"
            if not (vfile.exists() and vfile.stat().st_size > 0):
                print(f"[topo_fea] skip {name}: no interface file {vfile}")
                return

            vert_df = pd.read_csv(vfile, sep=' ', names=['ID','co_1','co_2','co_3'])
            res_list = list(vert_df['ID'])

            obj = topo_fea(str(pdb_path), NEI_DIS, E_SET, res_list, Cut=NEI_DIS)
            df = obj.cal_fea()
            (topo_dir / f"{name}.csv").write_text(df.to_csv(index=False))
        except Exception as e:
            print(f"[topo_fea] error in {name}: {e}")

    Parallel(n_jobs=jobs, backend="loky", batch_size=1, prefer="processes", verbose=0)(
    delayed(_one)(m) for m in model_names
    )


def _resolve_structure_path(name: str, complex_dir: Path) -> Optional[Path]:
    for ext in (".pdb", ".cif", ".mmcif"):
        p = complex_dir / f"{name}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

def run_node_features(model_names: Sequence[str], complex_dir: Path, topo_dir: Path,
                      interface_dir: Path, fea_dir: Path, jobs: int) -> None:
    """Calculate node features (basic + PH) in parallel."""
    # detect which constructor we have
    try:
        _sig = inspect.signature(node_fea)
        _nparams = len(_sig.parameters)
    except Exception:
        _nparams = None  # fall back to try/except per-call

    def _node_one(name: str) -> None:
        try:
            src = _resolve_structure_path(name, complex_dir)
            if src is None:
                print(f"[node_fea] skip {name}: no source PDB/mmCIF under {complex_dir}")
                return

            # Prefer passing FULL PATH if supported
            nf = None
            if _nparams == 3:
                # expected: node_fea(full_path, interface_dir, topo_dir)
                nf = node_fea(str(src), str(interface_dir), str(topo_dir))
            elif _nparams == 4:
                # expected: node_fea(name, pdb_dir, interface_dir, topo_dir)
                nf = node_fea(name, str(complex_dir), str(interface_dir), str(topo_dir))
            else:
                # unknown signature — try full path first, then (name, pdb_dir, ...)
                try:
                    nf = node_fea(str(src), str(interface_dir), str(topo_dir))
                except TypeError:
                    nf = node_fea(name, str(complex_dir), str(interface_dir), str(topo_dir))

            fea_df, _ = nf.calculate_fea()
            pd.set_option("future.no_silent_downcasting", True)
            fea_df.replace("NA", np.nan, inplace=True)
            fea_df = fea_df.dropna()
            (fea_dir / f"{name}.csv").write_text(fea_df.to_csv(index=False))
        except Exception as e:
            print(f"[node_fea] error in {name}: {e}")

    Parallel(n_jobs=jobs, backend="loky", batch_size=1, prefer="processes", verbose=0)(
    delayed(_node_one)(m) for m in model_names
    )

def run_graph_construction(model_names: Sequence[str], fea_dir: Path, interface_dir: Path,
                           arr_cutoff: Sequence[str], graph_dir: Path,
                           complex_dir: Path, jobs: int) -> None:
    """Build PyG graphs and store to disk, then validate each saved graph."""

    def _validate_graph_file(name: str) -> bool:
        """
        Try to load the saved graph and check the basics the model expects.
        Returns True if usable; False otherwise (and removes the bad file).
        """
        # The create_graph() code typically writes <name>.pt
        candidates = [
            graph_dir / f"{name}.pt",
            graph_dir / f"{name}.pth",
            graph_dir / f"{name}.bin",
        ]
        p = next((c for c in candidates if c.exists() and c.stat().st_size > 0), None)
        if p is None:
            print(f"[validate] {name}: no graph file produced.")
            return False

        try:
            g = torch.load(p, map_location="cpu")
            # Some save paths store a list; take the first element.
            if isinstance(g, list):
                if len(g) == 0:
                    raise ValueError("empty list")
                g = g[0]

            # Minimal fields the model & PyG ops rely on:
            missing = []
            for attr in ("x", "edge_index", "edge_attr"):
                if not hasattr(g, attr) or getattr(g, attr) is None:
                    missing.append(attr)
            if missing:
                raise ValueError(f"missing fields: {','.join(missing)}")

            # Sanity on shapes/dtypes
            if g.edge_index.dim() != 2 or g.edge_index.size(0) != 2:
                raise ValueError(f"bad edge_index shape: {tuple(g.edge_index.shape)}")
            if g.edge_attr.dim() != 2:
                raise ValueError(f"bad edge_attr shape: {tuple(g.edge_attr.shape)}")
            if g.edge_index.dtype != torch.long:
                # Fixup common pitfall
                g.edge_index = g.edge_index.long()
            # Some old artifacts may have a stale 'batch' attribute set to None
            if hasattr(g, "batch") and g.batch is None:
                delattr(g, "batch")

            # If we changed anything, rewrite to keep the fixed version on disk
            torch.save(g, p)
            return True

        except Exception as e:
            print(f"[validate] {name}: invalid graph ({e}). Removing.")
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            return False

    def _graph_one(name: str) -> None:
        try:
            create_graph(name, str(fea_dir), str(interface_dir), arr_cutoff, str(graph_dir), str(complex_dir))
            ok = _validate_graph_file(name)
            if not ok:
                print(f"[create_graph] {name}: validation failed; graph removed.")
        except Exception as e:
            print(f"[create_graph] error in {name}: {e}")

    Parallel(n_jobs=jobs, backend="loky", batch_size=1, prefer="processes", verbose=0)(
    delayed(_graph_one)(m) for m in model_names
    )

# ------------------------ Dataset / Inference ------------------------ #

class GraphDataset(Dataset):
    """Loads pre-built torch_geometric Data objects saved by create_graph()."""
    def __init__(self, graph_dir: Path, model_names: Sequence[str]):
        self.paths = []
        for name in model_names:
            candidates = [
                graph_dir / f"{name}.pt",
                graph_dir / f"{name}.pth",
                graph_dir / f"{name}.bin",
            ]
            for c in candidates:
                if c.exists():
                    self.paths.append(c)
                    break
        if not self.paths:
            raise FileNotFoundError(f"No graph files found in {graph_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):

        p = self.paths[idx]
        item = torch.load(p, map_location="cpu")

        # attach a stable decoy identifier
        try:
            setattr(item, 'name', p.stem)
        except Exception:
            pass

        return item


from torch_geometric.data import Batch, Data

def pyg_collate(data_list):
    cleaned = []
    for d in data_list:
        if hasattr(d, "batch"):
            try: delattr(d, "batch")
            except Exception: d.batch = None

        # Basic validity: nodes & edges present
        nn = getattr(d, "num_nodes", None)
        if nn is None and getattr(d, "x", None) is not None:
            nn = d.x.size(0)
        ei = getattr(d, "edge_index", None)
        has_nodes = (nn is not None and int(nn) > 0)
        has_edges = (ei is not None and ei.numel() > 0)
        if has_nodes and has_edges:
            cleaned.append(d)
        else:
            print("[warn] dropping invalid graph in batch: nodes/edges missing")

    if not cleaned:
        # Prevent Batch.from_data_list([]) crash; return an empty single-graph
        empty = Data()
        empty.edge_index = torch.empty((2,0), dtype=torch.long)
        empty.x = torch.empty((0,0))
        return Batch.from_data_list([empty])

    return Batch.from_data_list(cleaned)

def multi_pyg_collate(samples):
    """
    Collate function that supports:
      - samples = [Data, Data, ...]  -> returns a single Batch
      - samples = [[Data_i0, ..., Data_iL-1], ...] -> returns [Batch_0, ..., Batch_L-1]
    """
    if not samples:
        return []

    first = samples[0]

    # Case A: each sample is a (list|tuple) of Data, length = num_net
    if isinstance(first, (list, tuple)):
        L = len(first)
        out = []
        for i in range(L):
            data_list_i = []
            for s in samples:
                d = s[i]
                # Remove stale .batch (PyG will re-create it)
                if hasattr(d, "batch"):
                    try:
                        delattr(d, "batch")
                    except Exception:
                        d.batch = None
                data_list_i.append(d)
            out.append(Batch.from_data_list(data_list_i))
        return out

    # Case B: single Data per sample
    return pyg_collate(samples)

def load_model(checkpoint_path: Path, device: torch.device):
    chk = torch.load(str(checkpoint_path), map_location="cpu")
    model = GNN_edge1_edgepooling("mean", num_net=1, edge_dim=11, heads=8)
    state = chk["state_dict"] if isinstance(chk, dict) and "state_dict" in chk else chk
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

def _ensure_batch_vector(b):
    if getattr(b, "batch", None) is not None:
        return b
    if hasattr(b, "ptr") and b.ptr is not None:
        ptr = b.ptr
        counts = ptr[1:] - ptr[:-1]
        device = ptr.device
        b.batch = torch.arange(len(counts), device=device).repeat_interleave(counts)
        return b
    # Fallback build
    n = None
    if hasattr(b, "num_nodes") and b.num_nodes is not None:
        n = int(b.num_nodes)
    elif hasattr(b, "x") and b.x is not None:
        n = int(b.x.size(0))
    elif hasattr(b, "edge_index") and b.edge_index is not None and b.edge_index.numel() > 0:
        n = int(b.edge_index.max().item() + 1)
    if n is None:
        raise RuntimeError("Cannot determine number of nodes to build 'batch' vector.")
    b.batch = torch.zeros(n, dtype=torch.long, device=b.edge_index.device if hasattr(b, "edge_index") else None)
    return b


def infer_dataset_and_target(complex_folder: Path) -> tuple[str, str]:
    """
    Infer (dataset_tag, target_tag) from --complex-folder.

    Handles:
      - .../<DATASET>/decoy/<TARGET>
      - .../<DATASET>/<DATASET>_structures/<TARGET>
    Fallback (no recognized middle dir): dataset=parent.name, target=basename.
    """
    cf = complex_folder.resolve()
    target = cf.name
    parent = cf.parent
    pname = parent.name.lower()

    # Case 1: .../<DATASET>/decoy/<TARGET>
    if pname.startswith('decoy'):
        dataset = parent.parent.name if parent.parent is not None else parent.name
        return dataset, target

    # Case 2: .../<DATASET>/<DATASET>_structures/<TARGET>
    # Generalize for *_structures / *_structure / *_decoys / *_decoy
    suffixes = ('_structures', '_structure', '_decoys', '_decoy')
    if any(pname.endswith(sfx) for sfx in suffixes):
        grand = parent.parent
        if grand is not None and grand.name:
            dataset = grand.name
            return dataset, target

    # Fallback: use direct parent as dataset
    dataset = parent.name
    return dataset, target


def infer(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    preds: List[float] = []
    printed = False  # one-time debug print

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            # -- one-time debug (use first Batch if list) --
            if not printed:
                try:
                    n_batches = len(loader)
                except Exception:
                    n_batches = -1
                b0 = batch[0] if isinstance(batch, (list, tuple)) else batch
                try:
                    n_nodes = int(b0.num_nodes) if hasattr(b0, "num_nodes") else -1
                except Exception:
                    n_nodes = -1
                has_batch_vec = hasattr(b0, "batch") and (b0.batch is not None)
                print(f"[debug] infer(): batches={n_batches}, device={device}, "
                      f"first_batch_nodes={n_nodes}, has_batch={has_batch_vec}")
                if hasattr(b0, "edge_index"):
                    try:
                        ei = b0.edge_index
                        print(f"[debug] first_batch edge_index shape: {tuple(ei.size())}")
                    except Exception as e:
                        print(f"[debug] edge_index shape unavailable: {e}")
                printed = True

            # -- move to device & ensure .batch vectors exist --
            if isinstance(batch, (list, tuple)):
                batch = [ _ensure_batch_vector(b.to(device)) for b in batch ]
                model_input = batch
            else:
                b = _ensure_batch_vector(batch.to(device))
                model_input = [b]   # model expects a list

            # -- forward --
            out = model(model_input)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.detach().cpu().numpy().ravel()
            preds.extend(out.tolist())

    return np.asarray(preds, dtype=np.float32)


# ------------------------------ main ------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopoQA inference pipeline (refactored).")
    # add --dump-embeddings/--labels-csv/--outdir
    add_embedding_args(p)
    p.add_argument("--complex-folder", type=str, required=True,
                   help="Folder with input complex structures (.pdb/.cif).")
    p.add_argument("--work-dir", type=str, required=True,
                   help="Working directory for intermediate artifacts.")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Directory to write result CSV.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained GNN checkpoint (.pt/.pth).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (PyTorch). 0 = main process.")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                   help="Parallel jobs for feature/graph generation (joblib).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--cutoff", type=float, default=10.0,
                   help="Inter-chain Cα–Cα cutoff used by create_graph (Å).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute intermediates even if present.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"[info] device = {device}")

    complex_dir = Path(args.complex_folder).expanduser().resolve()
    work_dir    = Path(args.work_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    ckpt_path   = Path(args.checkpoint).expanduser().resolve()

    if not complex_dir.is_dir():
        print(f"[error] complex-folder does not exist: {complex_dir}")
        return 2
    if not any(complex_dir.glob("*.pdb")) and not any(complex_dir.glob("*.cif")):
        print(f"[error] no .pdb/.cif found in complex-folder: {complex_dir}")
        return 2
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}")
        return 2

    ensure_dir(work_dir)
    ensure_dir(results_dir)

    interface_dir = work_dir / "interface_ca"
    topo_dir      = work_dir / "node_topo"
    fea_dir       = work_dir / "node_fea"
    graph_dir     = work_dir / "graph"
    for d in (interface_dir, topo_dir, fea_dir, graph_dir):
        ensure_dir(d)

    # Determine model list deterministically
    all_models = list_models_from_dir(complex_dir)
    if not all_models:
        print(f"[error] No input complexes in {complex_dir}")
        return 2
    print(f"[info] found {len(all_models)} models in {complex_dir}")

    # Step 1: Interfaces
    if args.overwrite or not any(interface_dir.iterdir()):
        print(f"[stage] extracting interfaces -> {interface_dir} (jobs={args.jobs})")
        run_interface_extraction(complex_dir, interface_dir, args.jobs)
    else:
        print(f"[stage] reusing existing interfaces in {interface_dir}")

    # K - change
    # Keep models for which interface extraction succeeded
    iface_exts = (".txt", ".csv")  # <- your repo emits .txt here
    models_with_iface = sorted({
        p.stem
        for p in interface_dir.iterdir()
        if p.is_file() and p.suffix.lower() in iface_exts and p.stat().st_size > 0
})

    print(f"[debug] interface files found: {len(models_with_iface)}")
    if len(models_with_iface) == 0:
        print(f"[debug] sample of files in {interface_dir}:")
        for p in list(interface_dir.iterdir())[:10]:
            print(" -", p.name, p.stat().st_size, "bytes")

    model_names = sorted(set(all_models) & set(models_with_iface))
    if not model_names:
        print("[error] No models with detected interfaces. Nothing to do.")
        return 2
    print(f"[info] {len(model_names)} models with interfaces.")


    # K - change
    # ---- NEW robust validation 
    def _resolve_model_file(name: str, base: Path) -> Optional[Path]:
        """Return full path if a .pdb/.cif file exists for model `name`, else None."""
        for ext in (".pdb", ".cif"):
            candidate = base / f"{name}{ext}"
            if candidate.exists():
                return candidate
        return None

    missing = [m for m in model_names if _resolve_model_file(m, complex_dir) is None]
    if missing:
        print(f"[warn] {len(missing)} models skipped (no source PDB/mmCIF found): {missing[:5]} ...")
        model_names = [m for m in model_names if m not in missing]
    if not model_names:
        print("[error] No models left after validating source files.")
        return 2


    # Step 2: Topology features
    if args.overwrite or not any(topo_dir.iterdir()):
        print(f"[stage] computing topology features -> {topo_dir} (jobs={args.jobs})")
        run_topology_features(model_names, complex_dir, topo_dir, interface_dir, args.jobs)
    else:
        print(f"[stage] reusing existing topology in {topo_dir}")

    # Step 3: Node features
    if args.overwrite or not any(fea_dir.iterdir()):
        print(f"[stage] computing node features -> {fea_dir} (jobs={args.jobs})")
        run_node_features(model_names, complex_dir, topo_dir, interface_dir, fea_dir, args.jobs)
    else:
        print(f"[stage] reusing existing features in {fea_dir}")

    # Step 4: Graphs
    arr_cutoff = [f"0-{int(args.cutoff)}"]  # original API used ['0-10']
    if args.overwrite or not any(graph_dir.iterdir()):
        print(f"[stage] constructing graphs -> {graph_dir} (jobs={args.jobs}, cutoff={arr_cutoff[0]} Å)")
        run_graph_construction(model_names, fea_dir, interface_dir, arr_cutoff, graph_dir, complex_dir, args.jobs)

        # after run_graph_construction(...)
        good = list(Path(graph_dir).glob("*.pt")) + list(Path(graph_dir).glob("*.pth")) + list(Path(graph_dir).glob("*.bin"))
        print(f"[stage] graphs available for dataset: {len(good)}")

    else:
        print(f"[stage] reusing existing graphs in {graph_dir}")

    # Dataset / loader
    ds = GraphDataset(graph_dir, model_names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type != "cpu"), collate_fn=multi_pyg_collate)

    # Load model & infer
    print(f"[stage] loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path, device)
    
    # ===== Begin: pca, tsne embeddings wiring =====

    # ===== Begin: embeddings wiring (per-target) =====

    dataset_tag, target_tag = infer_dataset_and_target(complex_dir)
    labels_map = load_dockq_map(args.labels_csv, target_filter=target_tag)  # robust & filtered
    model_order = [p.stem for p in ds.paths]  # stable decoy ids in loader order

    if args.dump_embeddings != 'off':
        run_and_dump_embeddings(
            model=model,
            device=device,
            loader=loader,
            model_order=model_order,
            dataset_name=dataset_tag,
            target_name=target_tag,
            labels_map=labels_map,
            outdir=args.outdir,
            mode=args.dump_embeddings
        )

# ===== End: embeddings wiring =====
 

    # ===== End: embeddings wiring =====

    print(f"[stage] running inference on {len(ds)} graphs (batch_size={args.batch_size})")
    pred = infer(model, loader, device)

    # Results
    model_order = [p.stem for p in ds.paths]
    out_df = pd.DataFrame({"MODEL": model_order, "PRED_DOCKQ": pred})
    out_df = out_df.sort_values("PRED_DOCKQ", ascending=False).reset_index(drop=True)
    out_csv = results_dir / "result.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[done] wrote {len(out_df)} predictions to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

