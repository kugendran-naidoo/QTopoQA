"""
Core graph-building utilities that adapt existing TopoQA feature generators,
while allowing feature toggling via configs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric import data as pyg_data

from .config import EdgeConfig, NodeConfig
from .original_compat import inter_chain_dis, get_element_index_dis_atom, get_all_col


def _select_node_columns(df: pd.DataFrame, cfg: NodeConfig) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select columns from merged node features according to NodeConfig.
    - Assumes df has at least columns: ID, rasa, phi, psi, SS8_*, AA_*, topo f0_*, f1_*
    """
    cols: List[str] = []
    if cfg.use_rasa:
        cols.append("rasa")
    if cfg.use_phi_psi:
        cols.extend(["phi", "psi"])
    if cfg.use_ss8:
        cols.extend([f"SS8_{i}" for i in range(8)])
    if cfg.use_aa_onehot:
        cols.extend([f"AA_{i}" for i in range(21)])
    if cfg.use_topological:
        topo_cols = [c for c in df.columns if c.startswith("f0_") or c.startswith("f1_")]
        cols.extend(topo_cols)
    # ensure presence
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy(), keep


@dataclass
class GraphArtifacts:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    def to_pyg(self) -> pyg_data.Data:
        g = pyg_data.Data(
            x=self.x, edge_index=self.edge_index.long(), edge_attr=self.edge_attr
        )
        return g


def _minmax_scale(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    mn = a.min(axis=0, keepdims=True)
    mx = a.max(axis=0, keepdims=True)
    denom = np.where((mx - mn) == 0, 1.0, (mx - mn))
    return (a - mn) / denom


def build_graph(
    model_name: str,
    node_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    node_cfg: NodeConfig,
    edge_cfg: EdgeConfig,
    pdb_path: Optional[str] = None,
) -> GraphArtifacts:
    """
    Construct PyG graph tensors from node features and an interface coordinate table.

    interface_df is expected to have columns ['ID','co_1','co_2','co_3'] coordinated with node_df['ID'].
    node_df contains at least 'ID' and feature columns.
    """
    logger = logging.getLogger("qtdaqa")

    # Merge to ensure ordering and coordinate availability (preserve node_df order)
    if "ID" not in node_df.columns:
        raise ValueError("node_df must contain column 'ID'")
    vcols = ["ID", "co_1", "co_2", "co_3"]
    if not set(vcols).issubset(set(interface_df.columns)):
        raise ValueError("interface_df must have columns: ID, co_1, co_2, co_3")
    merged = pd.merge(node_df, interface_df[vcols], on="ID", how="left")
    if merged.empty:
        raise ValueError(f"no merged node/interface rows for {model_name}")

    # Select node feature columns per config; for legacy all_atom mode, enforce get_all_col order
    if (edge_cfg.mode or "").lower() == "all_atom":
        fea_col = get_all_col()
        x = torch.tensor(node_df[fea_col].values, dtype=torch.float32)
    else:
        x_df, kept = _select_node_columns(merged, node_cfg)
        x = torch.tensor(x_df.values, dtype=torch.float32)

    # Edge construction
    mode = (edge_cfg.mode or "simple").lower()
    if mode == "all_atom":
        if not pdb_path:
            raise ValueError("pdb_path is required for edge_cfg.mode='all_atom'")
        # Build vertice_df_filter like legacy: merge on ID, keep coords
        vertice_df_filter = merged[["ID", "co_1", "co_2", "co_3"]].copy()
        # Inclusivity: adjust cutoffs by a tiny epsilon to emulate inclusivity using legacy strict comparison
        eps = 1e-9
        min_val = float(edge_cfg.cutoff_min) - (eps if edge_cfg.cutoff_inclusive_min else 0.0)
        max_val = float(edge_cfg.cutoff_max) + (eps if edge_cfg.cutoff_inclusive_max else 0.0)
        arr_cut = [str(min_val), str(max_val)]

        # Compute adjacency and real distances using legacy routine
        dis, dis_real = inter_chain_dis.Calculate_distance(vertice_df_filter, arr_cut)

        # Load PDB model for all-atom histograms
        from Bio import PDB as _PDB
        parser = _PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]

        edge_index_list, edge_attr_np = get_element_index_dis_atom(dis_real, dis, 1.0, vertice_df_filter, model)
        if len(edge_index_list) == 0:
            logger.warning(f"{model_name}: no inter-chain edges in cutoff ({edge_cfg.cutoff_min},{edge_cfg.cutoff_max}) Å")
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous() if edge_index_list else torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32) if edge_attr_np.size > 0 else torch.zeros((0, 1), dtype=torch.float32)
    else:
        # Simple fast path: CA distances + radial bins
        coords = merged[["co_1", "co_2", "co_3"]].to_numpy(dtype=float)
        n = coords.shape[0]
        src, dst = [], []
        dists: List[float] = []
        cutoff_min = float(edge_cfg.cutoff_min)
        cutoff_max = float(edge_cfg.cutoff_max)
        chain_ids = merged["ID"].astype(str).str.extract(r"c<([^>]+)>")[0].fillna("")
        for i in range(n):
            for j in range(i + 1, n):
                if chain_ids.iat[i] == chain_ids.iat[j]:
                    continue
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                # inclusivity in simple mode
                lower_ok = dij > cutoff_min if not edge_cfg.cutoff_inclusive_min else dij >= cutoff_min
                upper_ok = dij < cutoff_max if not edge_cfg.cutoff_inclusive_max else dij <= cutoff_max
                if lower_ok and upper_ok:
                    src.extend([i, j])
                    dst.extend([j, i])
                    dists.extend([dij, dij])

        if not src:
            logger.warning(f"{model_name}: no inter-chain edges in cutoff [{cutoff_min},{cutoff_max}] Å")
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # Edge attributes per config
        e_attrs: List[List[float]] = []
        if len(dists) > 0:
            if edge_cfg.include_distance and not edge_cfg.include_histogram:
                e_attrs = [[d] for d in dists]
            elif edge_cfg.include_histogram:
                bins = np.linspace(1.0, 10.0, edge_cfg.histogram_bins)
                for d in dists:
                    hist = np.zeros(edge_cfg.histogram_bins, dtype=float)
                    idx = int(np.searchsorted(bins, min(d, bins[-1]), side="right") - 1)
                    idx = max(0, min(idx, edge_cfg.histogram_bins - 1))
                    hist[idx] += 1.0
                    row = ([d] if edge_cfg.include_distance else []) + hist.tolist()
                    e_attrs.append(row)
            else:
                e_attrs = [[0.0] for _ in dists]

        edge_attr = torch.tensor(np.asarray(e_attrs, dtype=float), dtype=torch.float32) if e_attrs else torch.zeros((0, 1), dtype=torch.float32)

        # Optional scaling
        if edge_attr.numel() > 0 and edge_cfg.scale.lower() == "minmax_sklearn":
            from sklearn.preprocessing import MinMaxScaler as _MMS
            mms = _MMS()
            edge_attr = torch.tensor(mms.fit_transform(edge_attr.numpy()), dtype=torch.float32)

    return GraphArtifacts(model_name, x=x, edge_index=edge_index, edge_attr=edge_attr)
