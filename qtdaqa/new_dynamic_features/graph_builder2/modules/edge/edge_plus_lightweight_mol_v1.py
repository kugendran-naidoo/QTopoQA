from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import numpy as np
import pandas as pd

from ...lib.edge_common import InterfaceResidue, StructureCache
from ...lib.laplacian_moments import LaplacianMomentConfig, build_unweighted_adjacency, compute_laplacian_moments
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
class EdgePlusLightweightMoLModule(EdgeFeatureModule):
    module_id = "edge/edge_plus_lightweight_MoL/v1"
    module_kind = "edge"
    default_alias = "Legacy 11D Edge + 5D unweighted Lap moments (mu1-3, kappa2, kappa3) on pair neighborhood = Edge 16D (lean MoL)"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Legacy histogram plus lean Laplacian moment context on the induced bipartite pair neighborhood (unweighted).",
        description=(
            "Prepends the legacy 11D distance histogram, then adds 5D unweighted normalized-Laplacian moments "
            "(mu1-3, kappa2, kappa3) on the induced bipartite neighborhood of each inter-chain edge (0-10 A window). "
            "Uses exact eigs when nodes <= size_threshold (80), otherwise SLQ (probes=8, steps=32). "
            "Deterministic ordering; optional CSV dumps."
        ),
        inputs=("interface_residues", "pdb_structure", "topology_csv"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "distance_min": "Minimum interface Cα distance (Å) to include an edge.",
            "distance_max": "Maximum interface Cα distance (Å) to include an edge.",
            "scale_histogram": "Apply MinMax scaling across the legacy distance + histogram block only.",
            "lap_size_threshold": "Node-count threshold for switching from exact eigs to SLQ.",
            "lap_estimator": "exact (default) or slq.",
            "lap_slq_probes": "Probe vectors for SLQ.",
            "lap_slq_steps": "Lanczos steps placeholder (kept for interface compatibility).",
            "lap_max_neighbors": "Cap on neighborhood size for Laplacian context (includes endpoints).",
            "lap_profile": "Profile Laplacian moment wall time (writes to metadata).",
            "jobs": "Optional override for edge assembly worker count.",
        },
        defaults={
            "distance_min": 0.0,
            "distance_max": 10.0,
            "scale_histogram": True,
            "lap_size_threshold": 80,
            "lap_estimator": "exact",
            "lap_slq_probes": 8,
            "lap_slq_steps": 32,
            "lap_max_neighbors": 128,
            "lap_profile": False,
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
        size_threshold = int(params.get("lap_size_threshold") or 80)
        estimator = str(params.get("lap_estimator") or "exact").strip().lower()
        slq_probes = int(params.get("lap_slq_probes") or 8)
        slq_steps = int(params.get("lap_slq_steps") or 32)
        max_neighbors = int(params.get("lap_max_neighbors") or 128)
        lap_profile = bool(params.get("lap_profile", False))
        jobs = params.get("jobs")

        if estimator not in {"exact", "slq"}:
            raise ValueError("edge.params.lap_estimator must be 'exact' or 'slq'.")

        config = LaplacianMomentConfig(
            size_threshold=size_threshold,
            estimator=estimator,
            slq_probes=slq_probes,
            slq_steps=slq_steps,
        )

        edges: List[List[int]] = []
        distances: List[float] = []
        hist_rows: List[List[float]] = []
        moments_rows: List[List[float]] = []

        chains = np.array([res.chain_id for res in residues])
        coords = np.stack([res.coord for res in residues], axis=0).astype(np.float64) if residues else np.empty((0, 3))

        lap_time = 0.0
        if len(residues) > 1:
            import time
            diff = coords[:, None, :] - coords[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            for i in range(len(residues)):
                for j in range(len(residues)):
                    if i == j or chains[i] == chains[j]:
                        continue
                    distance = float(dist_matrix[i, j])
                    if distance <= distance_min or distance >= distance_max:
                        continue
                    src_idx = id_to_index.get(residues[i].descriptor)
                    dst_idx = id_to_index.get(residues[j].descriptor)
                    if src_idx is None or dst_idx is None:
                        continue

                    residue_a = structure.get_residue(residues[i].chain_id, residues[i].residue_seq, residues[i].insertion_code)
                    residue_b = structure.get_residue(residues[j].chain_id, residues[j].residue_seq, residues[j].insertion_code)
                    if residue_a is None or residue_b is None:
                        continue
                    hist = _distance_histogram(residue_a, residue_b)
                    legacy_block = [distance] + hist.tolist()

                    neighborhood_indices = self._collect_neighborhood(residues, i, j, dist_matrix, distance_max, max_neighbors)
                    node_coords = [residues[idx].coord for idx in neighborhood_indices]
                    node_chains = [residues[idx].chain_id for idx in neighborhood_indices]
                    adj = build_unweighted_adjacency(node_coords, node_chains, distance_max)
                    lap_start = time.perf_counter()
                    raw, centered = compute_laplacian_moments(adj, moment_orders=(1, 2, 3), config=config)
                    lap_time += time.perf_counter() - lap_start
                    kappa2 = centered[0] if len(centered) > 0 else 0.0
                    kappa3 = centered[1] if len(centered) > 1 else 0.0
                    moments = [raw[0] if len(raw) > 0 else 0.0, raw[1] if len(raw) > 1 else 0.0, raw[2] if len(raw) > 2 else 0.0, kappa2, kappa3]

                    edges.append([src_idx, dst_idx])
                    distances.append(distance)
                    hist_rows.append(legacy_block)
                    moments_rows.append(moments)

                    edges.append([dst_idx, src_idx])
                    distances.append(distance)
                    hist_rows.append(legacy_block)
                    moments_rows.append(moments)

        if edges:
            hist_matrix = np.asarray(hist_rows, dtype=np.float32)
            if scale_histogram and hist_matrix.size:
                col_min = hist_matrix.min(axis=0, keepdims=True)
                col_max = hist_matrix.max(axis=0, keepdims=True)
                denom = np.where(col_max - col_min == 0.0, 1.0, col_max - col_min)
                hist_matrix = (hist_matrix - col_min) / denom

            moments_matrix = np.asarray(moments_rows, dtype=np.float32)
            feature_matrix = np.concatenate([hist_matrix, moments_matrix], axis=1) if moments_matrix.size else hist_matrix
            edge_index = np.asarray(edges, dtype=np.int64)

            order = np.lexsort((np.asarray(distances, dtype=np.float64), edge_index[:, 1], edge_index[:, 0]))
            edge_index = edge_index[order]
            feature_matrix = feature_matrix[order]
            distances = [distances[idx] for idx in order]
        else:
            feature_matrix = np.empty((0, self._HIST_DIM + 5), dtype=np.float32)
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
            "edge_feature_variant": "edge_plus_lightweight_MoL/lean",
            "distance_window": [float(distance_min), float(distance_max)],
            "histogram_dim": self._HIST_DIM,
            "lap_size_threshold": size_threshold,
            "lap_estimator": estimator,
            "lap_slq_probes": slq_probes,
            "lap_slq_steps": slq_steps,
            "lap_max_neighbors": max_neighbors,
        }
        if lap_profile:
            metadata["lap_profile"] = True
            metadata["lap_time_sec"] = float(lap_time)
        if jobs is not None:
            metadata["requested_jobs"] = int(jobs)

        return EdgeBuildResult(edge_index=edge_index, edge_attr=feature_matrix, metadata=metadata)

    def _collect_neighborhood(
        self,
        residues: List[InterfaceResidue],
        i: int,
        j: int,
        dist_matrix: np.ndarray,
        cutoff: float,
        max_neighbors: int,
    ) -> List[int]:
        """Collect endpoints and opposite-chain neighbors within cutoff."""
        included = {i, j}
        for idx, res in enumerate(residues):
            if res.chain_id == residues[i].chain_id or res.chain_id == residues[j].chain_id:
                continue
            if dist_matrix[i, idx] <= cutoff or dist_matrix[j, idx] <= cutoff:
                included.add(idx)
            if len(included) >= max_neighbors:
                break
        return sorted(included)

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

        size_threshold = params.get("lap_size_threshold")
        if size_threshold is not None:
            params["lap_size_threshold"] = require_positive_int(size_threshold, "edge.params.lap_size_threshold")
        estimator = params.get("lap_estimator")
        if estimator is not None:
            est = str(estimator).strip().lower()
            if est not in {"exact", "slq"}:
                raise ValueError("edge.params.lap_estimator must be 'exact' or 'slq'.")
            params["lap_estimator"] = est
        slq_probes = params.get("lap_slq_probes")
        if slq_probes is not None:
            params["lap_slq_probes"] = require_positive_int(slq_probes, "edge.params.lap_slq_probes")
        slq_steps = params.get("lap_slq_steps")
        if slq_steps is not None:
            params["lap_slq_steps"] = require_positive_int(slq_steps, "edge.params.lap_slq_steps")
        max_neighbors = params.get("lap_max_neighbors")
        if max_neighbors is not None:
            params["lap_max_neighbors"] = require_positive_int(max_neighbors, "edge.params.lap_max_neighbors")
        lap_profile = params.get("lap_profile")
        if lap_profile is not None:
            params["lap_profile"] = require_bool(lap_profile, "edge.params.lap_profile")

        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "edge.params.jobs")

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        base = super().config_template()
        base["alias"] = cls.default_alias
        base["summary"] = cls._metadata.summary
        base["description"] = cls._metadata.description
        param_comments = dict(base.get("param_comments", {}))
        param_comments.setdefault("distance_min", "Cα distance window start; edges with distance <= min are skipped")
        param_comments.setdefault("distance_max", "Cα distance window end; edges with distance >= max are skipped")
        param_comments.setdefault("scale_histogram", "Scale only the legacy 11D (distance + 10-bin) block across each graph")
        param_comments.setdefault("lap_size_threshold", "Node-count threshold for switching from exact eigs to SLQ (default 80)")
        param_comments.setdefault("lap_estimator", "exact (default) or slq")
        param_comments.setdefault("lap_slq_probes", "Probe vectors for SLQ (default 8)")
        param_comments.setdefault("lap_slq_steps", "Lanczos steps placeholder (default 32; kept for compatibility)")
        param_comments.setdefault("lap_max_neighbors", "Cap neighborhood size for Laplacian context (includes endpoints)")
        param_comments.setdefault("lap_profile", "Profile Laplacian moment wall time (writes to metadata)")
        param_comments.setdefault("jobs", "Honors CLI --jobs > config default_jobs > module jobs; edge ordering deterministic; dumps resorted by edge_runner")
        base["param_comments"] = param_comments
        base["dim_hint"] = "# dim: 11 legacy histogram + 5 Lap moments = 16 (lean)"
        return base
