from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
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


@register_feature_module
class EdgePlusMinAggTopoModule(EdgeFeatureModule):
    module_id = "edge/edge_plus_min_agg_topo/v1"
    module_kind = "edge"
    default_alias = "11D Legacy + 434D (lean) / 561D (heavy) 10A"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Legacy 11D histogram plus per-residue topology concat+abs-diff+cosine (lean).",
        description=(
            "Reuses per-residue topology vectors to build edge features without rerunning persistence. "
            "For each cross-chain edge, concatenates (u_topo, v_topo, |u-v|) and appends cosine similarity "
            "and optional endpoint norms. Legacy 11D histogram (distance + 10-bin atom distance histogram) "
            "is prepended. Deterministic edge ordering is preserved (src_idx, dst_idx, distance)."
        ),
        inputs=("interface_residues", "pdb_structure", "topology_csv"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "distance_min": "Minimum interface Cα distance (Å) to include an edge.",
            "distance_max": "Maximum interface Cα distance (Å) to include an edge.",
            "scale_histogram": "Apply MinMax scaling across the legacy distance + histogram block only.",
            "include_norms": "Include L2 norms of endpoint topology vectors.",
            "include_cosine": "Include cosine similarity between endpoint topology vectors.",
            "include_minmax": "Include per-dimension min/max across endpoints (used by heavy variant).",
            "variant": "Aggregation variant: lean (default) or heavy (adds min/max block).",
            "jobs": "Optional override for edge assembly worker count.",
        },
        defaults={
            "distance_min": 0.0,
            "distance_max": 10.0,
            "scale_histogram": True,
            "include_norms": True,
            "include_cosine": True,
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
        include_minmax = params.get("include_minmax", False)
        variant = params.get("variant", "lean")
        jobs = params.get("jobs")

        if variant not in {"lean", "heavy"}:
            raise ValueError(f"Unsupported variant '{variant}' for edge_plus_min_agg_topo (expected lean|heavy).")

        topo_map, topo_dim = _load_topology_vectors(topology_path)
        if topo_dim <= 0:
            raise ValueError(f"No topology feature columns found in {topology_path}")
        minmax_dim = topo_dim * 2 if (variant == "heavy" and include_minmax) else 0
        agg_dim = topo_dim * 3 + minmax_dim + (2 if include_norms else 0) + (1 if include_cosine else 0)

        edges: List[List[int]] = []
        distances: List[float] = []
        hist_rows: List[List[float]] = []
        agg_rows: List[np.ndarray] = []

        for i, src in enumerate(residues):
            src_idx = id_to_index.get(src.descriptor)
            if src_idx is None:
                continue
            for j, dst in enumerate(residues):
                if i == j or src.chain_id == dst.chain_id:
                    continue
                dst_idx = id_to_index.get(dst.descriptor)
                if dst_idx is None:
                    continue
                distance = float(np.linalg.norm(dst.coord - src.coord))
                if distance <= distance_min or distance >= distance_max:
                    continue

                residue_a = structure.get_residue(src.chain_id, src.residue_seq, src.insertion_code)
                residue_b = structure.get_residue(dst.chain_id, dst.residue_seq, dst.insertion_code)
                if residue_a is None or residue_b is None:
                    continue

                topo_src = topo_map.get(src.descriptor)
                topo_dst = topo_map.get(dst.descriptor)
                if topo_src is None or topo_dst is None:
                    raise ValueError(f"Missing topology vector for {src.descriptor} or {dst.descriptor}")

                hist = _distance_histogram(residue_a, residue_b)
                legacy_block = [distance] + hist.tolist()

                parts: List[np.ndarray] = [topo_src, topo_dst, np.abs(topo_src - topo_dst)]
                if variant == "heavy" and include_minmax:
                    parts.append(np.minimum(topo_src, topo_dst))
                    parts.append(np.maximum(topo_src, topo_dst))
                if include_norms:
                    parts.append(np.array([np.linalg.norm(topo_src), np.linalg.norm(topo_dst)], dtype=np.float32))
                if include_cosine:
                    parts.append(np.array([_safe_cosine(topo_src, topo_dst)], dtype=np.float32))
                agg_vector = np.concatenate(parts, axis=0).astype(np.float32)

                for edge_pair in ((src_idx, dst_idx), (dst_idx, src_idx)):
                    edges.append([edge_pair[0], edge_pair[1]])
                    distances.append(distance)
                    hist_rows.append(legacy_block)
                    agg_rows.append(agg_vector.copy())

        if edges:
            order = sorted(range(len(edges)), key=lambda idx: (edges[idx][0], edges[idx][1], distances[idx]))
            ordered_edges = [edges[idx] for idx in order]
            ordered_hist = [hist_rows[idx] for idx in order]
            ordered_agg = [agg_rows[idx] for idx in order]

            hist_matrix = np.asarray(ordered_hist, dtype=np.float32)
            if scale_histogram and hist_matrix.size:
                col_min = hist_matrix.min(axis=0, keepdims=True)
                col_max = hist_matrix.max(axis=0, keepdims=True)
                denom = np.where(col_max - col_min == 0.0, 1.0, col_max - col_min)
                hist_matrix = (hist_matrix - col_min) / denom

            agg_matrix = np.vstack(ordered_agg).astype(np.float32) if ordered_agg else np.empty((0, agg_dim))
            feature_matrix = np.concatenate([hist_matrix, agg_matrix], axis=1) if ordered_agg else hist_matrix
            edge_index = np.asarray(ordered_edges, dtype=np.int64)
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
                for (src_idx, dst_idx), dist in zip(edges, distances):
                    writer.writerow([src_idx, dst_idx, dist])

        feature_dim = int(feature_matrix.shape[1]) if feature_matrix.ndim >= 2 else 0
        metadata: Dict[str, Any] = {
            "edge_count": int(edge_index.shape[0]),
            "feature_dim": feature_dim,
            "edge_feature_variant": "edge_plus_min_agg_topo/lean",
            "variant": variant,
            "topology_feature_dim": topo_dim,
            "include_norms": bool(include_norms),
            "include_cosine": bool(include_cosine),
            "include_minmax": bool(include_minmax) if variant == "heavy" else False,
            "distance_window": [float(distance_min), float(distance_max)],
            "histogram_dim": self._HIST_DIM,
            "aggregation_dim": agg_dim,
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
        base_params = dict(base.get("params", {}))
        param_comments = dict(base.get("param_comments", {}))
        param_comments.setdefault("include_minmax", "heavy variant only; adds per-dimension min/max blocks")
        param_comments.setdefault("variant", "lean or heavy")
        base["param_comments"] = param_comments

        heavy_params = dict(base_params)
        heavy_params.update({"variant": "heavy", "include_minmax": True})
        base["alternates"] = [
            {
                "module": cls.module_id,
                "alias": "11D Legacy + 434D (lean) / 561D (heavy) 10A",
                "params": heavy_params,
                "param_comments": param_comments,
                "summary": cls._metadata.summary,
                "description": cls._metadata.description,
            }
        ]
        return base
