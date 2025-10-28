from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from ..lib.edge_common import InterfaceResidue, StructureCache
except ImportError:  # pragma: no cover
    from lib.edge_common import InterfaceResidue, StructureCache  # type: ignore
from .base import EdgeBuildResult, EdgeFeatureModule, build_metadata
from .registry import register_feature_module


@dataclass
class EdgeBand:
    label: str
    min_distance: float
    max_distance: float


def _compute_histogram(coords_a: np.ndarray, coords_b: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if coords_a.size == 0 or coords_b.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
    hist, _ = np.histogram(distances, bins=bins)
    total = hist.sum()
    if total > 0:
        return hist.astype(float) / float(total)
    return hist.astype(float)


def _encode_chain(chain_id: str) -> float:
    if not chain_id:
        return 0.0
    base = ord("A")
    value = ord(chain_id.upper()[0]) - base + 1
    return max(0.0, min(1.0, value / 26.0))


def _encode_residue(res_name: str) -> float:
    amino_acids = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    ]
    aa_index = {name: idx for idx, name in enumerate(amino_acids)}
    idx = aa_index.get(res_name.upper(), -1)
    if idx < 0:
        return 1.0
    return (idx + 1) / (len(amino_acids) + 1)


class MultiscaleEdgeBuilder:
    def __init__(self, params: Dict[str, object]):
        self.params = params
        bands_cfg = params.get("bands") or [
            {"label": "short", "min_distance": 0.0, "max_distance": 6.0},
            {"label": "medium", "min_distance": 6.0, "max_distance": 10.0},
            {"label": "long", "min_distance": 10.0, "max_distance": 14.0},
        ]
        self.bands: List[EdgeBand] = [
            EdgeBand(label=band["label"], min_distance=float(band["min_distance"]), max_distance=float(band["max_distance"]))
            for band in bands_cfg
        ]
        hist_bins = params.get("histogram_bins") or [float(x) for x in range(0, 22, 2)]
        self.hist_bins = np.asarray(hist_bins, dtype=float)
        self.contact_threshold = float(params.get("contact_threshold", 5.0))
        self.include_inverse_distance = bool(params.get("include_inverse_distance", True))
        self.include_unit_vector = bool(params.get("include_unit_vector", True))
        self.unit_vector_epsilon = float(params.get("unit_vector_epsilon", 1e-8))

    def build(
        self,
        residues: List[InterfaceResidue],
        id_to_index: Dict[str, int],
        structure: StructureCache,
        dump_path: Optional[Path] = None,
    ) -> EdgeBuildResult:
        feature_rows: List[List[float]] = []
        edge_index: List[List[int]] = []
        dump_rows: List[Dict[str, object]] = []

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
                vector = dst.coord - src.coord
                distance = float(np.linalg.norm(vector))
                band_idx = self._band_index(distance)
                if band_idx is None:
                    continue
                features_forward = self._assemble_features(distance, vector, src, dst, band_idx, structure)
                features_reverse = self._assemble_features(distance, -vector, dst, src, band_idx, structure)
                feature_rows.append(features_forward)
                feature_rows.append(features_reverse)
                edge_index.append([src_idx, dst_idx])
                edge_index.append([dst_idx, src_idx])
                if dump_path is not None:
                    dump_rows.append(
                        {
                            "src_idx": src_idx,
                            "dst_idx": dst_idx,
                            "src_id": src.descriptor,
                            "dst_id": dst.descriptor,
                            "band": self.bands[band_idx].label,
                            "distance": distance,
                        }
                    )

        if not feature_rows:
            feature_matrix = np.empty((0, 0), dtype=np.float32)
            edge_array = np.empty((0, 2), dtype=np.int64)
        else:
            feature_matrix = np.asarray(feature_rows, dtype=np.float32)
            edge_array = np.asarray(edge_index, dtype=np.int64)

        if dump_path is not None and dump_rows:
            dump_df = pd.DataFrame(dump_rows)
            dump_df.to_csv(dump_path, index=False)

        metadata = {
            "edge_count": int(edge_array.shape[0]),
            "feature_dim": int(feature_matrix.shape[1] if feature_matrix.size else 0),
            "bands": [band.label for band in self.bands],
        }

        return EdgeBuildResult(edge_index=edge_array, edge_attr=feature_matrix, metadata=metadata)

    def _band_index(self, distance: float) -> Optional[int]:
        for idx, band in enumerate(self.bands):
            if band.min_distance <= distance < band.max_distance:
                return idx
        return None

    def _assemble_features(
        self,
        distance: float,
        vector: np.ndarray,
        src: InterfaceResidue,
        dst: InterfaceResidue,
        band_idx: int,
        structure: StructureCache,
    ) -> List[float]:
        eps = self.unit_vector_epsilon
        features: List[float] = [distance]
        if self.include_inverse_distance:
            features.append(1.0 / max(distance, eps))
        if self.include_unit_vector:
            if distance > eps:
                unit_vec = vector / distance
            else:
                unit_vec = np.zeros(3, dtype=float)
            features.extend(unit_vec.tolist())

        features.append(_encode_chain(src.chain_id))
        features.append(_encode_chain(dst.chain_id))
        features.append(_encode_residue(src.residue_name))
        features.append(_encode_residue(dst.residue_name))
        features.append(float(band_idx))

        band_one_hot = [1.0 if idx == band_idx else 0.0 for idx in range(len(self.bands))]
        features.extend(band_one_hot)

        residue_a = structure.get_residue(src.chain_id, src.residue_seq, src.insertion_code)
        residue_b = structure.get_residue(dst.chain_id, dst.residue_seq, dst.insertion_code)
        coords_a = _atom_coordinates(residue_a) if residue_a is not None else np.empty((0, 3))
        coords_b = _atom_coordinates(residue_b) if residue_b is not None else np.empty((0, 3))
        hist = _compute_histogram(coords_a, coords_b, self.hist_bins)
        features.extend(hist.tolist())

        if coords_a.size and coords_b.size:
            diffs = coords_a[:, None, :] - coords_b[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
            contact_count = np.count_nonzero(distances <= self.contact_threshold)
            features.append(float(contact_count))
        else:
            features.append(0.0)

        return features


def _atom_coordinates(residue) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        coord = atom.get_coord()
        coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


@register_feature_module
class MultiscaleEdgeModuleV24(EdgeFeatureModule):
    module_id = "edge/multi_scale/v24"
    module_kind = "edge"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="24-D multi-scale edge features with distance bands and histograms.",
        description=(
            "Computes directed residue-residue edges with distance, inverse distance, "
            "unit vectors, chain/amino-acid encodings, band indicators, inter-residue "
            "distance histograms, and contact counts."
        ),
        inputs=("interface_residues", "node_features", "pdb_structure"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "bands": "Distance band definitions with labels, min, and max distances.",
            "histogram_bins": "Histogram bin edges (Å) for inter-atomic distances.",
            "contact_threshold": "Distance threshold (Å) for counting contacts.",
            "include_inverse_distance": "Include inverse distance feature.",
            "include_unit_vector": "Include unit vector components.",
            "unit_vector_epsilon": "Epsilon to avoid division by zero in unit vector calc.",
        },
        defaults={
            "bands": [
                {"label": "short", "min_distance": 0.0, "max_distance": 6.0},
                {"label": "medium", "min_distance": 6.0, "max_distance": 10.0},
                {"label": "long", "min_distance": 10.0, "max_distance": 14.0},
            ],
            "histogram_bins": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "contact_threshold": 5.0,
            "include_inverse_distance": True,
            "include_unit_vector": True,
            "unit_vector_epsilon": 1e-8,
        },
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.builder = MultiscaleEdgeBuilder(self.params)

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
        result = self.builder.build(residues, id_to_index, structure, dump_path=dump_path)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "model_key": model_key,
                "edge_feature_variant": "multi_scale_v24",
            }
        )
        return EdgeBuildResult(
            edge_index=result.edge_index,
            edge_attr=result.edge_attr,
            metadata=metadata,
        )
