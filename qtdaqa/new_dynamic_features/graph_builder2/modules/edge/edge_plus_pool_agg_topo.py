from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import heapq
import numpy as np
import pandas as pd

from ...lib.edge_common import InterfaceResidue, StructureCache
from ..base import (
    EdgeBuildResult,
    EdgeFeatureModule,
    build_metadata,
    require_bool,
    require_float,
    require_positive_int,
)
from ..registry import register_feature_module


def _atom_coordinates(residue) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        coord = atom.get_coord()
        coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


def _distance_histogram(residue_a, residue_b) -> np.ndarray:
    coords_a = _atom_coordinates(residue_a)
    coords_b = _atom_coordinates(residue_b)
    if coords_a.size == 0 or coords_b.size == 0:
        return np.zeros(10, dtype=float)
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
    bins = np.arange(1, 11, 1)
    bins = np.append(bins, np.inf)
    hist, _ = np.histogram(distances, bins=bins)
    return hist.astype(float)


def _load_topology_vectors(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"Topology file {path} missing ID column.")
    feature_cols = [col for col in df.columns if col != "ID"]
    vectors = df[feature_cols].to_numpy(dtype=np.float32)
    identifiers = df["ID"].astype(str).tolist()
    topo_map = {identifier: vector for identifier, vector in zip(identifiers, vectors)}
    return topo_map, len(feature_cols)


def _safe_cosine(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    denom = norm_a * norm_b
    if denom <= eps:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _k_nearest_residues(
    residues: List[InterfaceResidue],
    target: InterfaceResidue,
    *,
    k: int,
) -> List[InterfaceResidue]:
    heap: List[Tuple[float, str, InterfaceResidue]] = []
    for candidate in residues:
        if candidate.descriptor == target.descriptor:
            continue
        dist = float(np.linalg.norm(candidate.coord - target.coord))
        # use descriptor as a deterministic tie-breaker
        heapq.heappush(heap, (dist, candidate.descriptor, candidate))
    selected: List[InterfaceResidue] = []
    while heap and len(selected) < k:
        _, _, res = heapq.heappop(heap)
        selected.append(res)
    return selected


def _pooled_mean(
    residues: List[InterfaceResidue],
    topo_map: Dict[str, np.ndarray],
    target: InterfaceResidue,
    *,
    k: int,
    topo_dim: int,
) -> np.ndarray:
    neighbors = _k_nearest_residues(residues, target, k=k)
    if not neighbors:
        return np.zeros((topo_dim,), dtype=np.float32)
    vectors = []
    for res in neighbors:
        vec = topo_map.get(res.descriptor)
        if vec is None:
            raise ValueError(f"Missing topology vector for pooled neighbor {res.descriptor}")
        vectors.append(vec)
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32) if vectors else np.zeros((topo_dim,), dtype=np.float32)


def _compute_pooled_means_vectorized(
    residues: List[InterfaceResidue],
    topo_vectors: np.ndarray,
    descriptors: List[str],
    *,
    k: int,
) -> np.ndarray:
    """Compute pooled neighbor means for each residue using vectorized distance ranking.

    Tie-breaks for equal distances use descriptor rank (lexicographic) to stay deterministic.
    Missing topo vectors should already be filtered before calling this helper.
    """
    n = len(residues)
    if n == 0:
        return np.empty((0, topo_vectors.shape[1]), dtype=np.float32)

    coords = np.stack([res.coord for res in residues], axis=0).astype(np.float64)
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist_matrix, np.inf)

    # Tie-breaker: add a tiny offset proportional to descriptor rank (lexicographic).
    rank = {desc: idx for idx, desc in enumerate(sorted(descriptors))}
    rank_vec = np.array([rank[d] for d in descriptors], dtype=np.float64)
    dist_with_ties = dist_matrix + (rank_vec[None, :] * 1e-9)

    # Mask any missing topo vectors (should not occur, but keep defensive).
    missing = np.isnan(topo_vectors).any(axis=1)
    if missing.any():
        dist_with_ties[:, missing] = np.inf

    neighbors = np.argsort(dist_with_ties, axis=1)[:, :k]
    pooled = np.zeros_like(topo_vectors, dtype=np.float32)
    for i in range(n):
        idxs = neighbors[i]
        finite_mask = np.isfinite(dist_matrix[i, idxs])
        if not finite_mask.any():
            continue
        pooled[i] = topo_vectors[idxs[finite_mask]].mean(axis=0).astype(np.float32)
    return pooled


@register_feature_module
class EdgePlusPoolAggTopoModule(EdgeFeatureModule):
    module_id = "edge/edge_plus_pool_agg_topo/v1"
    module_kind = "edge"
    default_alias = "Legacy 11D Edge + 1126D {(hist + endpoint agg + pooled agg) from 140D PH} = Edge 1137D (Lean) | Legacy 11D Edge + 1686D {(hist + endpoint agg + pooled agg + minmax) from 140D PH} = Edge 1697D (Heavy)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Legacy 11D histogram plus balanced topo agg with pooled neighbor means (lean).",
        description=(
            "Extends balanced aggregation by adding pooled neighbor topology means per endpoint (k-nearest), "
            "then computing mean/abs-diff/cosine (and optional norms) on both endpoints and pooled summaries. "
            "Legacy 11D histogram (distance + 10-bin atom distance histogram) is prepended. Deterministic "
            "edge ordering is preserved (src_idx, dst_idx, distance)."
        ),
        inputs=("interface_residues", "pdb_structure", "topology_csv"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "distance_min": "Minimum interface Cα distance (Å) to include an edge.",
            "distance_max": "Maximum interface Cα distance (Å) to include an edge.",
            "scale_histogram": "Apply MinMax scaling across the legacy distance + histogram block only.",
            "include_norms": "Include L2 norms of endpoint topology vectors (and pooled means).",
            "include_cosine": "Include cosine similarity between endpoint (and pooled) topology vectors.",
            "pool_k": "Number of nearest interface residues to include in pooled means per endpoint.",
            "include_minmax": "Include per-dimension min/max across endpoints and pooled summaries (heavy variant).",
            "variant": "Aggregation variant: lean (default) or heavy (adds min/max blocks).",
            "jobs": "Optional override for edge assembly worker count.",
        },
        defaults={
            "distance_min": 0.0,
            "distance_max": 10.0,
            "scale_histogram": True,
            "include_norms": True,
            "include_cosine": True,
            "pool_k": 5,
            "include_minmax": False,
            "variant": "lean",
            "jobs": 16,
        },
    )

    _HIST_DIM = 11  # distance + 10 histogram bins

    def build_edges(
        self,
        *,
        model_key: str,
        residues: List[InterfaceResidue],
        id_to_index: Dict[str, int],
        structure: StructureCache,
        node_df: pd.DataFrame,
        interface_path: Path,
        topology_path: Path,
        node_path: Path,
        pdb_path: Path,
        dump_path: Optional[Path] = None,
    ) -> EdgeBuildResult:
        params = self.params
        distance_min = params.get("distance_min", 0.0)
        distance_max = params.get("distance_max", 10.0)
        scale_histogram = params.get("scale_histogram", True)
        include_norms = params.get("include_norms", True)
        include_cosine = params.get("include_cosine", True)
        pool_k = params.get("pool_k", 5)
        variant = params.get("variant", "lean")
        include_minmax = params.get("include_minmax", False)
        jobs = params.get("jobs")

        if variant not in {"lean", "heavy"}:
            raise ValueError(f"Unsupported variant '{variant}' for edge_plus_pool_agg_topo (expected lean|heavy).")

        topo_map, topo_dim = _load_topology_vectors(topology_path)
        if topo_dim <= 0:
            raise ValueError(f"No topology feature columns found in {topology_path}")

        k_neighbors = max(1, int(pool_k)) if pool_k is not None else 5
        minmax_dim = topo_dim * 2 if (variant == "heavy" and include_minmax) else 0
        endpoint_agg_dim = topo_dim * 4 + minmax_dim + (2 if include_norms else 0) + (1 if include_cosine else 0)
        pooled_agg_dim = topo_dim * 4 + minmax_dim + (2 if include_norms else 0) + (1 if include_cosine else 0)
        agg_dim = endpoint_agg_dim + pooled_agg_dim

        # Prepare arrays
        descriptors = [res.descriptor for res in residues]
        topo_vectors = []
        valid_residue = []
        for res in residues:
            vec = topo_map.get(res.descriptor)
            topo_vectors.append(vec if vec is not None else np.full((topo_dim,), np.nan, dtype=np.float32))
            valid_residue.append(vec is not None and id_to_index.get(res.descriptor) is not None)
        topo_vectors = np.stack(topo_vectors, axis=0).astype(np.float32) if topo_vectors else np.empty((0, topo_dim), dtype=np.float32)
        valid_mask = np.array(valid_residue, dtype=bool)

        coords = np.stack([res.coord for res in residues], axis=0).astype(np.float64) if residues else np.empty((0, 3))
        chains = np.array([res.chain_id for res in residues])

        # Compute pooled means (vectorized) for valid residues only
        pooled_means = _compute_pooled_means_vectorized(residues, topo_vectors, descriptors, k=k_neighbors)

        edges: List[List[int]] = []
        distances: List[float] = []
        hist_rows: List[List[float]] = []
        src_pos_list: List[int] = []
        dst_pos_list: List[int] = []

        if len(residues) > 1:
            diff = coords[:, None, :] - coords[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            for i in range(len(residues)):
                if not valid_mask[i]:
                    continue
                for j in range(len(residues)):
                    if i == j or chains[i] == chains[j] or not valid_mask[j]:
                        continue
                    distance = float(dist_matrix[i, j])
                    if distance <= distance_min or distance >= distance_max:
                        continue

                    residue_a = structure.get_residue(residues[i].chain_id, residues[i].residue_seq, residues[i].insertion_code)
                    residue_b = structure.get_residue(residues[j].chain_id, residues[j].residue_seq, residues[j].insertion_code)
                    if residue_a is None or residue_b is None:
                        continue

                    hist = _distance_histogram(residue_a, residue_b)
                    legacy_block = [distance] + hist.tolist()

                    src_idx = id_to_index.get(residues[i].descriptor)
                    dst_idx = id_to_index.get(residues[j].descriptor)
                    if src_idx is None or dst_idx is None:
                        continue

                    # Emit both directions to preserve legacy behavior.
                    edges.append([src_idx, dst_idx])
                    distances.append(distance)
                    hist_rows.append(legacy_block)
                    src_pos_list.append(i)
                    dst_pos_list.append(j)

                    edges.append([dst_idx, src_idx])
                    distances.append(distance)
                    hist_rows.append(legacy_block)
                    src_pos_list.append(j)
                    dst_pos_list.append(i)

        if edges:
            src_pos = np.array(src_pos_list, dtype=np.int64)
            dst_pos = np.array(dst_pos_list, dtype=np.int64)

            topo_src = topo_vectors[src_pos].astype(np.float32, copy=False)
            topo_dst = topo_vectors[dst_pos].astype(np.float32, copy=False)
            pooled_src = pooled_means[src_pos].astype(np.float32, copy=False)
            pooled_dst = pooled_means[dst_pos].astype(np.float32, copy=False)

            mean_vec = (topo_src + topo_dst) * 0.5
            pooled_mean_vec = (pooled_src + pooled_dst) * 0.5
            abs_diff = np.abs(topo_src - topo_dst)
            pooled_abs_diff = np.abs(pooled_src - pooled_dst)

            parts_endpoint = [topo_src, topo_dst, mean_vec, abs_diff]
            parts_pooled = [pooled_src, pooled_dst, pooled_mean_vec, pooled_abs_diff]

            if variant == "heavy" and include_minmax:
                parts_endpoint.extend([np.minimum(topo_src, topo_dst), np.maximum(topo_src, topo_dst)])
                parts_pooled.extend([np.minimum(pooled_src, pooled_dst), np.maximum(pooled_src, pooled_dst)])

            if include_norms:
                endpoint_norms = np.stack(
                    [np.linalg.norm(topo_src, axis=1), np.linalg.norm(topo_dst, axis=1)],
                    axis=1,
                ).astype(np.float32)
                pooled_norms = np.stack(
                    [np.linalg.norm(pooled_src, axis=1), np.linalg.norm(pooled_dst, axis=1)],
                    axis=1,
                ).astype(np.float32)
                parts_endpoint.append(endpoint_norms)
                parts_pooled.append(pooled_norms)

            if include_cosine:
                eps = 1e-8
                dot_endpoint = np.einsum("ij,ij->i", topo_src, topo_dst)
                norm_src = np.linalg.norm(topo_src, axis=1)
                norm_dst = np.linalg.norm(topo_dst, axis=1)
                denom = norm_src * norm_dst
                cosine_endpoint = np.where(denom > eps, dot_endpoint / denom, 0.0).astype(np.float32)
                parts_endpoint.append(cosine_endpoint[:, None])

                dot_pooled = np.einsum("ij,ij->i", pooled_src, pooled_dst)
                norm_p_src = np.linalg.norm(pooled_src, axis=1)
                norm_p_dst = np.linalg.norm(pooled_dst, axis=1)
                denom_p = norm_p_src * norm_p_dst
                cosine_pooled = np.where(denom_p > eps, dot_pooled / denom_p, 0.0).astype(np.float32)
                parts_pooled.append(cosine_pooled[:, None])

            endpoint_vector = np.concatenate(parts_endpoint, axis=1).astype(np.float32, copy=False)
            pooled_vector = np.concatenate(parts_pooled, axis=1).astype(np.float32, copy=False)
            agg_matrix = np.concatenate([endpoint_vector, pooled_vector], axis=1).astype(np.float32, copy=False)

            hist_matrix = np.asarray(hist_rows, dtype=np.float32)
            if scale_histogram and hist_matrix.size:
                col_min = hist_matrix.min(axis=0, keepdims=True)
                col_max = hist_matrix.max(axis=0, keepdims=True)
                denom = np.where(col_max - col_min == 0.0, 1.0, col_max - col_min)
                hist_matrix = (hist_matrix - col_min) / denom

            feature_matrix = np.concatenate([hist_matrix, agg_matrix], axis=1) if agg_matrix.size else hist_matrix
            edge_index = np.asarray(edges, dtype=np.int64)

            order = np.lexsort((np.asarray(distances, dtype=np.float64), edge_index[:, 1], edge_index[:, 0]))
            edge_index = edge_index[order]
            feature_matrix = feature_matrix[order]
            distances = [distances[idx] for idx in order]
        else:
            hist_matrix = np.empty((0, self._HIST_DIM), dtype=np.float32)
            agg_matrix = np.empty((0, agg_dim), dtype=np.float32)
            feature_matrix = np.empty((0, self._HIST_DIM + agg_dim), dtype=np.float32)
            edge_index = np.empty((0, 2), dtype=np.int64)

        if dump_path is not None and edges:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["src_idx", "dst_idx", "distance"])
                for (src_idx, dst_idx), dist in zip(edge_index.tolist(), distances):
                    writer.writerow([src_idx, dst_idx, dist])

        feature_dim = int(feature_matrix.shape[1]) if feature_matrix.ndim >= 2 else 0
        metadata: Dict[str, Any] = {
            "edge_count": int(edge_index.shape[0]),
            "feature_dim": feature_dim,
            "edge_feature_variant": "edge_plus_pool_agg_topo/lean",
            "variant": variant,
            "topology_feature_dim": topo_dim,
            "include_norms": bool(include_norms),
            "include_cosine": bool(include_cosine),
            "include_minmax": bool(include_minmax) if variant == "heavy" else False,
            "pool_k": k_neighbors,
            "distance_window": [float(distance_min), float(distance_max)],
            "histogram_dim": self._HIST_DIM,
            "aggregation_dim": agg_dim,
            "endpoint_aggregation_dim": endpoint_agg_dim,
            "pooled_aggregation_dim": pooled_agg_dim,
        }
        if jobs is not None:
            metadata["requested_jobs"] = int(jobs)

        return EdgeBuildResult(edge_index=edge_index, edge_attr=feature_matrix, metadata=metadata)

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        distance_min = params.get("distance_min")
        distance_max = params.get("distance_max")
        if distance_min is not None:
            params["distance_min"] = require_float(distance_min, "edge.params.distance_min")
        if distance_max is not None:
            params["distance_max"] = require_float(distance_max, "edge.params.distance_max")
        if distance_min is not None and distance_max is not None:
            if params["distance_min"] >= params["distance_max"]:
                raise ValueError("edge.params.distance_min must be < distance_max.")

        scale_histogram = params.get("scale_histogram")
        if scale_histogram is not None:
            params["scale_histogram"] = require_bool(scale_histogram, "edge.params.scale_histogram")

        include_norms = params.get("include_norms")
        if include_norms is not None:
            params["include_norms"] = require_bool(include_norms, "edge.params.include_norms")

        include_cosine = params.get("include_cosine")
        if include_cosine is not None:
            params["include_cosine"] = require_bool(include_cosine, "edge.params.include_cosine")

        pool_k = params.get("pool_k")
        if pool_k is not None:
            params["pool_k"] = require_positive_int(pool_k, "edge.params.pool_k")

        include_minmax = params.get("include_minmax")
        if include_minmax is not None:
            params["include_minmax"] = require_bool(include_minmax, "edge.params.include_minmax")

        variant = params.get("variant")
        if variant is not None:
            normalised = str(variant).strip().lower()
            if normalised not in {"lean", "heavy"}:
                raise ValueError("edge.params.variant must be 'lean' or 'heavy'.")
            params["variant"] = normalised

        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "edge.params.jobs")

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        base = super().config_template()
        param_comments = {
            "variant": "lean or heavy",
            "include_minmax": "heavy variant only; adds per-dimension min/max blocks to endpoint and pooled agg",
            "pool_k": "number of nearest interface residues to pool per endpoint (default 5)",
        }
        params = dict(cls._metadata.defaults)
        heavy_params = dict(params)
        heavy_params.update({"variant": "heavy", "include_minmax": True})
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "alternates": [
                {
                    "module": cls.module_id,
                    "alias": cls.default_alias,
                    "params": heavy_params,
                    "param_comments": param_comments,
                    "summary": cls._metadata.summary,
                    "description": cls._metadata.description,
                }
            ],
        }
