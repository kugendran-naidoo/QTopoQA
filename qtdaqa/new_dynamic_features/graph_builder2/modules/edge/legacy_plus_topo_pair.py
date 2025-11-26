from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import gudhi

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


def _match_residue(structure: StructureCache, residue: InterfaceResidue):
    return structure.get_residue(residue.chain_id, residue.residue_seq, residue.insertion_code)


def _select_neighbor_descriptors(
    residues: Sequence[InterfaceResidue],
    src: InterfaceResidue,
    dst: InterfaceResidue,
    neighbor_distance: float,
) -> List[InterfaceResidue]:
    if neighbor_distance <= 0:
        return []
    neighbors: List[InterfaceResidue] = []
    src_coord = src.coord
    dst_coord = dst.coord
    seen: Set[str] = {src.descriptor, dst.descriptor}
    for candidate in residues:
        if candidate.descriptor in seen:
            continue
        dist_src = np.linalg.norm(candidate.coord - src_coord)
        dist_dst = np.linalg.norm(candidate.coord - dst_coord)
        if dist_src <= neighbor_distance or dist_dst <= neighbor_distance:
            neighbors.append(candidate)
            seen.add(candidate.descriptor)
    return neighbors


def _collect_point_cloud(
    structure: StructureCache,
    src: InterfaceResidue,
    dst: InterfaceResidue,
    residues: Sequence[InterfaceResidue],
    include_neighbors: bool,
    neighbor_distance: float,
) -> np.ndarray:
    point_sets = []
    for residue in (src, dst):
        struct_res = _match_residue(structure, residue)
        if struct_res is not None:
            point_sets.append(_atom_coordinates(struct_res))
    if include_neighbors and neighbor_distance > 0:
        for neighbor in _select_neighbor_descriptors(residues, src, dst, neighbor_distance):
            struct_res = _match_residue(structure, neighbor)
            if struct_res is not None:
                point_sets.append(_atom_coordinates(struct_res))
    if not point_sets:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(point_sets, axis=0)


def _persistence_summary_zero_dim(
    persistence: Sequence[Tuple[int, Tuple[float, float]]],
    *,
    cutoff: float,
    min_persistence: float,
) -> np.ndarray:
    stats = np.zeros(5, dtype=float)
    filtered = [
        death - birth
        for dim, (birth, death) in persistence
        if dim == 0 and death != float("inf") and death <= cutoff and (death - birth) >= min_persistence
    ]
    if filtered:
        data = np.array(filtered, dtype=float)
        stats[:] = [
            float(data.sum()),
            float(data.min()),
            float(data.max()),
            float(data.mean()),
            float(data.std(ddof=0)),
        ]
    return stats


def _persistence_summary_one_dim(
    persistence: Sequence[Tuple[int, Tuple[float, float]]],
    *,
    min_persistence: float,
) -> np.ndarray:
    stats = np.zeros(15, dtype=float)
    filtered = [
        (birth, death)
        for dim, (birth, death) in persistence
        if dim == 1 and death != float("inf") and (death - birth) >= min_persistence
    ]
    if filtered:
        births = np.array([birth for birth, _ in filtered], dtype=float)
        deaths = np.array([death for _, death in filtered], dtype=float)
        spans = deaths - births
        stats[:] = [
            float(len(filtered)),
            float(births.sum()),
            float(births.min()),
            float(births.max()),
            float(births.mean()),
            float(births.std(ddof=0)),
            float(deaths.sum()),
            float(deaths.min()),
            float(deaths.max()),
            float(deaths.mean()),
            float(deaths.std(ddof=0)),
            float(spans.sum()),
            float(spans.min()),
            float(spans.max()),
            float(spans.mean()),
        ]
    return stats


def _compute_pair_topology_features(
    points: np.ndarray,
    filtration_cutoff: float,
    min_persistence: float,
) -> np.ndarray:
    if points.shape[0] < 2:
        return np.zeros(20, dtype=np.float32)
    try:
        rips = gudhi.RipsComplex(points=points, max_edge_length=filtration_cutoff)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()
    except Exception:
        return np.zeros(20, dtype=np.float32)
    stats0 = _persistence_summary_zero_dim(persistence, cutoff=filtration_cutoff, min_persistence=min_persistence)
    stats1 = _persistence_summary_one_dim(persistence, min_persistence=min_persistence)
    return np.concatenate([stats0, stats1], axis=0).astype(np.float32)


@register_feature_module
class LegacyPlusTopoPairEdgeModule(EdgeFeatureModule):
    module_id = "edge/legacy_plus_topo_pair"
    module_kind = "edge"
    default_alias = "Legacy 11D + pair topology"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="11-D legacy edges augmented with pair-wise topological summaries.",
        description=(
            "Extends the legacy histogram features by computing persistent homology summaries over the "
            "two residues forming each edge (optionally including nearby residues)."
        ),
        inputs=("interface_residues", "pdb_structure"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "distance_min": "Minimum interface Cα distance (Å) to include an edge.",
            "distance_max": "Maximum interface Cα distance (Å) to include an edge.",
            "scale_features": "Apply MinMax scaling across legacy histogram features.",
            "neighbor_distance": "Å radius for including additional interface residues around the pair (0 disables).",
            "include_neighbors": "When true, include residues within neighbor_distance of either endpoint.",
            "filtration_cutoff": "Maximum edge length considered when building the Rips complex (Å).",
            "min_persistence": "Minimum persistence used when summarising homology classes.",
            "jobs": "Optional override for edge assembly worker count.",
        },
        defaults={
            "distance_min": 0.0,
            "distance_max": 10.0,
            "scale_features": True,
            "neighbor_distance": 4.0,
            "include_neighbors": True,
            "filtration_cutoff": 8.0,
            "min_persistence": 0.01,
            "jobs": 16,
        },
    )

    _TOPO_DIM = 20

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
        neighbor_distance = params.get("neighbor_distance", 0.0)
        include_neighbors = params.get("include_neighbors", True)
        filtration_cutoff = params.get("filtration_cutoff", 8.0)
        min_persistence = params.get("min_persistence", 0.01)
        jobs = params.get("jobs")

        edges: List[List[int]] = []
        legacy_features: List[List[float]] = []
        topo_features: List[np.ndarray] = []

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
                residue_a = _match_residue(structure, src)
                residue_b = _match_residue(structure, dst)
                if residue_a is None or residue_b is None:
                    continue
                hist = _distance_histogram(residue_a, residue_b)
                feature = [distance] + hist.tolist()
                points = _collect_point_cloud(
                    structure,
                    src,
                    dst,
                    residues,
                    include_neighbors=include_neighbors,
                    neighbor_distance=float(neighbor_distance),
                )
                topo_vec = _compute_pair_topology_features(
                    points,
                    float(filtration_cutoff),
                    float(min_persistence),
                )
                legacy_features.append(feature)
                topo_features.append(topo_vec)
                legacy_features.append(feature.copy())
                topo_features.append(topo_vec.copy())
                edges.append([src_idx, dst_idx])
                edges.append([dst_idx, src_idx])

        if legacy_features:
            legacy_matrix = np.asarray(legacy_features, dtype=np.float32)
            if scale_features and legacy_matrix.size:
                col_min = legacy_matrix.min(axis=0, keepdims=True)
                col_max = legacy_matrix.max(axis=0, keepdims=True)
                denom = np.where(col_max - col_min == 0.0, 1.0, col_max - col_min)
                legacy_matrix = (legacy_matrix - col_min) / denom
            topo_matrix = np.vstack(topo_features).astype(np.float32)
            combined = np.concatenate([legacy_matrix, topo_matrix], axis=1)
            edge_index = np.asarray(edges, dtype=np.int64)
        else:
            combined = np.empty((0, 11 + self._TOPO_DIM), dtype=np.float32)
            edge_index = np.empty((0, 2), dtype=np.int64)

        if dump_path is not None and edges:
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("w", encoding="utf-8", newline="") as handle:
                import csv

                writer = csv.writer(handle)
                writer.writerow(["src_idx", "dst_idx", "distance"])
                for (src_idx, dst_idx), feature in zip(edges, legacy_features):
                    writer.writerow([src_idx, dst_idx, feature[0]])

        feature_dim = int(combined.shape[1]) if combined.ndim >= 2 else 0
        metadata = {
            "edge_count": int(edge_index.shape[0]),
            "feature_dim": feature_dim,
            "edge_feature_variant": "legacy_plus_topo_pair",
            "distance_window": [distance_min, distance_max],
            "topology_neighbor_distance": float(neighbor_distance),
            "topology_include_neighbors": bool(include_neighbors),
            "topology_filtration_cutoff": float(filtration_cutoff),
            "topology_min_persistence": float(min_persistence),
            "legacy_feature_dim": 11,
            "topology_feature_dim": self._TOPO_DIM,
        }
        if jobs is not None:
            metadata["requested_jobs"] = int(jobs)

        return EdgeBuildResult(edge_index=edge_index, edge_attr=combined, metadata=metadata)

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
        neighbor_distance = params.get("neighbor_distance")
        if neighbor_distance is not None:
            params["neighbor_distance"] = max(0.0, require_float(neighbor_distance, "edge.params.neighbor_distance"))
        include_neighbors = params.get("include_neighbors")
        if include_neighbors is not None:
            params["include_neighbors"] = require_bool(include_neighbors, "edge.params.include_neighbors")
        filtration_cutoff = params.get("filtration_cutoff")
        if filtration_cutoff is not None:
            params["filtration_cutoff"] = require_float(filtration_cutoff, "edge.params.filtration_cutoff")
        min_persistence = params.get("min_persistence")
        if min_persistence is not None:
            params["min_persistence"] = max(0.0, require_float(min_persistence, "edge.params.min_persistence"))
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "edge.params.jobs")
