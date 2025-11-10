from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...lib.edge_common import InterfaceResidue, StructureCache
from ..base import (
    EdgeBuildResult,
    EdgeFeatureModule,
    build_metadata,
    ensure_sorted_float_sequence,
    require_bool,
    require_float,
    require_positive_float,
)
from ..registry import register_feature_module


def _atom_coordinates(residue) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        coord = atom.get_coord()
        coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


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


@dataclass
class EdgeBand:
    label: str
    min_distance: float
    max_distance: float


class NeoEdgeBuilder:
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
        histogram_bins = params.get("histogram_bins") or [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        self.hist_bins = np.asarray(histogram_bins, dtype=float)
        legacy_bins = params.get("legacy_histogram_bins") or [0.0] + [float(x) for x in range(1, 11)]
        if legacy_bins[-1] <= legacy_bins[-2]:
            legacy_bins.append(legacy_bins[-1] + 0.0001)
        self.legacy_bins = np.asarray(legacy_bins, dtype=float)
        self.histogram_mode = str(params.get("histogram_mode", "density_times_contact"))
        thresholds = params.get("contact_thresholds") or [5.0, 10.0]
        if isinstance(thresholds, (list, tuple)):
            self.contact_thresholds = sorted(float(value) for value in thresholds)
        else:
            self.contact_thresholds = [float(thresholds)]
        self.include_inverse_distance = bool(params.get("include_inverse_distance", True))
        self.include_unit_vector = bool(params.get("include_unit_vector", True))
        self.unit_vector_epsilon = float(params.get("unit_vector_epsilon", 1e-8))
        self.scale_features = bool(params.get("scale_features", True))
        self.contact_normalizer = float(params.get("contact_normalizer", 50.0))
        self.short_contact_max = float(params.get("short_contact_max", self.contact_thresholds[0]))
        self.long_band_mask = bool(params.get("long_band_mask", True))
        self.max_distance = max(band.max_distance for band in self.bands)
        self.feature_dim = self._calculate_feature_dim()

    def _calculate_feature_dim(self) -> int:
        dim = 1  # distance
        if self.include_inverse_distance:
            dim += 1
        if self.include_unit_vector:
            dim += 3
        dim += 2  # chain encodings
        dim += 2  # residue encodings
        dim += 1  # band index
        dim += len(self.bands)  # band one-hot
        hist_len = len(self.legacy_bins) - 1
        if self.histogram_mode == "density_and_count":
            dim += hist_len * 2
        else:
            dim += hist_len
        dim += len(self.contact_thresholds)
        dim += 1  # short contact fraction
        if self.long_band_mask:
            dim += 1
        return dim

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
            feature_matrix = np.empty((0, self.feature_dim), dtype=np.float32)
            edge_array = np.empty((0, 2), dtype=np.int64)
        else:
            feature_matrix = np.asarray(feature_rows, dtype=np.float32)
            edge_array = np.asarray(edge_index, dtype=np.int64)
            if feature_matrix.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Computed edge feature dimension {feature_matrix.shape[1]} does not match expected {self.feature_dim}"
                )

        if dump_rows:
            path = dump_path or Path.cwd() / "edge_dumps" / "neo_edges.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            dump_df = pd.DataFrame(dump_rows)
            dump_df.to_csv(path, index=False)

        metadata = {
            "edge_count": int(edge_array.shape[0]),
            "feature_dim": int(self.feature_dim),
            "bands": [band.label for band in self.bands],
            "histogram_bins": self.hist_bins.tolist(),
            "legacy_histogram_bins": self.legacy_bins.tolist(),
            "histogram_mode": self.histogram_mode,
            "contact_thresholds": self.contact_thresholds,
            "scale_features": self.scale_features,
            "include_inverse_distance": self.include_inverse_distance,
            "include_unit_vector": self.include_unit_vector,
            "long_band_mask": self.long_band_mask,
            "contact_normalizer": self.contact_normalizer,
            "short_contact_max": self.short_contact_max,
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
        features: List[float] = []
        dist_value = distance / max(self.max_distance, 1e-8) if self.scale_features else distance
        features.append(dist_value)
        if self.include_inverse_distance:
            inv = 1.0 / max(distance, eps)
            if self.scale_features:
                inv = min(inv, self.max_distance / max(eps, 1e-8))
            features.append(inv)
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
        legacy_hist = self._histogram_from_bins(coords_a, coords_b, self.legacy_bins)
        hist_vector = self._prepare_histogram_vector(legacy_hist)
        features.extend(hist_vector.tolist())

        contact_counts = self._contact_counts(coords_a, coords_b)
        if self.scale_features:
            contact_counts = [count / max(self.contact_normalizer, 1e-8) for count in contact_counts]
        features.extend(contact_counts)

        coarse_hist = self._histogram_from_bins(coords_a, coords_b, self.hist_bins)
        short_fraction = self._short_contact_fraction(coarse_hist)
        features.append(short_fraction)
        if self.long_band_mask:
            long_only = 1.0 if (band_idx == len(self.bands) - 1 and short_fraction == 0.0) else 0.0
            features.append(long_only)
        return features

    def _histogram_from_bins(self, coords_a: np.ndarray, coords_b: np.ndarray, bins: np.ndarray) -> np.ndarray:
        if coords_a.size == 0 or coords_b.size == 0:
            return np.zeros(len(bins) - 1, dtype=float)
        diffs = coords_a[:, None, :] - coords_b[None, :, :]
        distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
        hist, _ = np.histogram(distances, bins=bins)
        return hist.astype(float)

    def _prepare_histogram_vector(self, hist: np.ndarray) -> np.ndarray:
        total = hist.sum()
        if total == 0:
            return np.zeros_like(hist)
        if self.histogram_mode == "density":
            return hist / float(total)
        if self.histogram_mode == "density_and_count":
            density = hist / float(total)
            return np.concatenate([density, hist])
        # density_times_contact -> emphasise magnitude via raw counts (hybrid approach)
        return hist

    def _short_contact_fraction(self, coarse_hist: np.ndarray) -> float:
        if coarse_hist.size == 0:
            return 0.0
        total = coarse_hist.sum()
        if total == 0:
            return 0.0
        upper_edges = self.hist_bins[1:]
        mask = upper_edges <= self.short_contact_max
        if not np.any(mask):
            return 0.0
        return float(coarse_hist[mask].sum() / total)

    def _contact_counts(self, coords_a: np.ndarray, coords_b: np.ndarray) -> List[float]:
        counts: List[float] = []
        if coords_a.size == 0 or coords_b.size == 0:
            return [0.0 for _ in self.contact_thresholds]
        diffs = coords_a[:, None, :] - coords_b[None, :, :]
        distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
        for threshold in self.contact_thresholds:
            counts.append(float(np.count_nonzero(distances <= threshold)))
        return counts


@register_feature_module
class NeoEdgeModuleV24(EdgeFeatureModule):
    module_id = "edge/neo/v24"
    module_kind = "edge"
    default_alias = "Neo hybrid multi-scale"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="27-D neo edge features (hybrid histogram + geometric cues).",
        description=(
            "Extends the legacy 11-D histogram with multi-scale distance bands, unit vectors,"
            " weighted histograms truncated to 10 Å, multiple contact thresholds, and optional long-band masks."
        ),
        inputs=("interface_residues", "node_features", "pdb_structure"),
        outputs=("edge_index", "edge_attr"),
        parameters={
            "bands": "Distance band definitions with labels, min, and max distances.",
            "histogram_bins": "Histogram bin edges (Å) truncated to 0–10 Å by default.",
            "legacy_histogram_bins": "High-resolution bins used for hybrid weighting.",
            "histogram_mode": "density | density_times_contact | density_and_count.",
            "contact_thresholds": "List of Å thresholds for contact counts (default [5,10]).",
            "include_inverse_distance": "Include inverse distance feature.",
            "include_unit_vector": "Include unit vector components.",
            "unit_vector_epsilon": "Epsilon to avoid division by zero in unit vector calc.",
            "scale_features": "Apply light feature scaling (distance/contact counts).",
            "contact_normalizer": "Value used to scale contact counts into [0,1].",
            "short_contact_max": "Upper Å bound for short-contact fraction feature.",
            "long_band_mask": "Append a mask so long-band edges with zero short contacts can be ignored.",
        },
        defaults={
            "bands": [
                {"label": "short", "min_distance": 0.0, "max_distance": 6.0},
                {"label": "medium", "min_distance": 6.0, "max_distance": 10.0},
                {"label": "long", "min_distance": 10.0, "max_distance": 14.0},
            ],
            "histogram_bins": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "histogram_mode": "density_times_contact",
            "contact_thresholds": [5.0, 10.0],
            "include_inverse_distance": True,
            "include_unit_vector": True,
            "unit_vector_epsilon": 1e-8,
            "scale_features": True,
            "contact_normalizer": 50.0,
            "short_contact_max": 5.0,
            "long_band_mask": True,
        },
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.builder = NeoEdgeBuilder(self.params)

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
                "edge_feature_variant": "neo_v24",
            }
        )
        return EdgeBuildResult(
            edge_index=result.edge_index,
            edge_attr=result.edge_attr,
            metadata=metadata,
        )

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        bands = params.get("bands")
        if bands is not None:
            if not isinstance(bands, (list, tuple)) or not bands:
                raise ValueError("edge.params.bands must be a non-empty list of band definitions.")
            normalised: List[Dict[str, Any]] = []
            for idx, band in enumerate(bands):
                if not isinstance(band, dict):
                    raise ValueError("Each entry in edge.params.bands must be a mapping.")
                label = band.get("label")
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("Each band definition must include a non-empty 'label'.")
                min_distance = require_float(
                    band.get("min_distance"),
                    f"edge.params.bands[{idx}].min_distance",
                )
                max_distance = require_float(
                    band.get("max_distance"),
                    f"edge.params.bands[{idx}].max_distance",
                )
                if min_distance >= max_distance:
                    raise ValueError("Each band must satisfy min_distance < max_distance.")
                normalised.append(
                    {
                        "label": label.strip(),
                        "min_distance": min_distance,
                        "max_distance": max_distance,
                    }
                )
            params["bands"] = normalised

        histogram_bins = params.get("histogram_bins")
        if histogram_bins is not None:
            params["histogram_bins"] = list(
                ensure_sorted_float_sequence(histogram_bins, "edge.params.histogram_bins")
            )

        legacy_bins = params.get("legacy_histogram_bins")
        if legacy_bins is not None:
            params["legacy_histogram_bins"] = list(
                ensure_sorted_float_sequence(
                    legacy_bins, "edge.params.legacy_histogram_bins", allow_equal=True
                )
            )

        histogram_mode = params.get("histogram_mode")
        if histogram_mode is not None:
            allowed = {"density", "density_times_contact", "density_and_count"}
            if histogram_mode not in allowed:
                raise ValueError(
                    "edge.params.histogram_mode must be one of "
                    f"{', '.join(sorted(allowed))}."
                )

        contact_thresholds = params.get("contact_thresholds")
        if contact_thresholds is not None:
            if not isinstance(contact_thresholds, (list, tuple)) or not contact_thresholds:
                raise ValueError("edge.params.contact_thresholds must be a non-empty list.")
            floats = [
                require_positive_float(value, "edge.params.contact_thresholds")
                for value in contact_thresholds
            ]
            params["contact_thresholds"] = sorted(floats)

        for key in ("include_inverse_distance", "include_unit_vector", "scale_features", "long_band_mask"):
            value = params.get(key)
            if value is not None:
                params[key] = require_bool(value, f"edge.params.{key}")

        unit_vector_epsilon = params.get("unit_vector_epsilon")
        if unit_vector_epsilon is not None:
            params["unit_vector_epsilon"] = require_positive_float(
                unit_vector_epsilon, "edge.params.unit_vector_epsilon"
            )

        contact_normalizer = params.get("contact_normalizer")
        if contact_normalizer is not None:
            params["contact_normalizer"] = require_positive_float(
                contact_normalizer, "edge.params.contact_normalizer", allow_zero=False
            )

        short_contact_max = params.get("short_contact_max")
        if short_contact_max is not None:
            params["short_contact_max"] = require_positive_float(
                short_contact_max, "edge.params.short_contact_max", allow_zero=False
            )
