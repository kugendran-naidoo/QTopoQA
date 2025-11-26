from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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


@register_feature_module
class LegacyEdgeModuleV11(EdgeFeatureModule):
    module_id = "edge/legacy_band/v11"
    module_kind = "edge"
    default_alias = "11D TopoQA default 10A cutoff"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="11-D legacy edge features (distance + atom distance histogram).",
        description=(
            "Reproduces the original 11-dimensional edge attributes consisting of "
            "residue pair distance and a 10-bin histogram of inter-atomic distances, "
            "scaled to [0, 1]."
        ),
        inputs=("interface_residues", "pdb_structure"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "distance_min": "Minimum interface Cα distance (Å) to include an edge.",
            "distance_max": "Maximum interface Cα distance (Å) to include an edge.",
            "scale_features": "Apply MinMax scaling across edge features.",
            "jobs": "Optional override for edge assembly worker count.",
        },
        defaults={"distance_min": 0.0, "distance_max": 10.0, "scale_features": True, "jobs": 16},
    )

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
        scale_features = params.get("scale_features", True)
        jobs = params.get("jobs")
        edges: List[List[int]] = []
        features: List[List[float]] = []

        for i, src in enumerate(residues):
            if src.chain_id is None:
                continue
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
                hist = _distance_histogram(residue_a, residue_b)
                feature = [distance] + hist.tolist()
                features.append(feature)
                features.append(feature.copy())
                edges.append([src_idx, dst_idx])
                edges.append([dst_idx, src_idx])

        if features:
            order = sorted(
                range(len(edges)),
                key=lambda idx: (edges[idx][0], edges[idx][1], features[idx][0]),
            )
            ordered_edges = [edges[idx] for idx in order]
            ordered_features = [features[idx] for idx in order]

            feature_matrix = np.asarray(ordered_features, dtype=np.float32)
            if scale_features and feature_matrix.size:
                col_min = feature_matrix.min(axis=0, keepdims=True)
                col_max = feature_matrix.max(axis=0, keepdims=True)
                denom = np.where(col_max - col_min == 0.0, 1.0, col_max - col_min)
                feature_matrix = (feature_matrix - col_min) / denom
            edge_index = np.asarray(ordered_edges, dtype=np.int64)
        else:
            feature_matrix = np.empty((0, 11), dtype=np.float32)
            edge_index = np.empty((0, 2), dtype=np.int64)

        if dump_path is not None and edges:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["src_idx", "dst_idx", "distance"])
                for (src_idx, dst_idx), feature in zip(edges, features):
                    writer.writerow([src_idx, dst_idx, feature[0]])

        feature_dim = int(feature_matrix.shape[1]) if feature_matrix.ndim >= 2 else 0
        metadata = {
            "edge_count": int(edge_index.shape[0]),
            "feature_dim": feature_dim,
            "edge_feature_variant": "legacy_v11",
            "distance_window": [distance_min, distance_max],
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
        scale_features = params.get("scale_features")
        if scale_features is not None:
            params["scale_features"] = require_bool(scale_features, "edge.params.scale_features")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "edge.params.jobs")
