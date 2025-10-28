from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from ..lib.edge_common import InterfaceResidue, StructureCache
except ImportError:  # pragma: no cover
    from lib.edge_common import InterfaceResidue, StructureCache  # type: ignore
from .base import EdgeBuildResult, EdgeFeatureModule, build_metadata
from .registry import register_feature_module


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


@dataclass
class LegacyEdgeParams:
    distance_min: float = 0.0
    distance_max: float = 10.0
    scale_features: bool = True


@register_feature_module
class LegacyEdgeModuleV11(EdgeFeatureModule):
    module_id = "edge/legacy_band/v11"
    module_kind = "edge"
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
        },
        defaults={"distance_min": 0.0, "distance_max": 10.0, "scale_features": True},
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
        params = LegacyEdgeParams(**self.params)
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
                if distance <= params.distance_min or distance >= params.distance_max:
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
            feature_matrix = np.asarray(features, dtype=np.float32)
            if params.scale_features:
                scaler = MinMaxScaler()
                feature_matrix = scaler.fit_transform(feature_matrix)
            edge_index = np.asarray(edges, dtype=np.int64)
        else:
            feature_matrix = np.empty((0, 11), dtype=np.float32)
            edge_index = np.empty((0, 2), dtype=np.int64)

        if dump_path is not None and edges:
            dump_df = pd.DataFrame(
                {
                    "src_idx": [e[0] for e in edges],
                    "dst_idx": [e[1] for e in edges],
                    "distance": [row[0] for row in features],
                }
            )
            dump_df.to_csv(dump_path, index=False)

        metadata = {
            "edge_count": int(edge_index.shape[0]),
            "feature_dim": int(feature_matrix.shape[1] if feature_matrix.size else 0),
            "edge_feature_variant": "legacy_v11",
            "distance_window": [params.distance_min, params.distance_max],
        }

        return EdgeBuildResult(edge_index=edge_index, edge_attr=feature_matrix, metadata=metadata)
