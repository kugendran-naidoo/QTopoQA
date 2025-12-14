#!/usr/bin/env python3
"""Topological feature extraction for protein structures.

This module replaces ``topo_feature.py`` with a cleaner implementation that is
easier to read, debug, and extend.  The key improvements are:

* dataclass-driven configuration instead of hard-coded constants;
* modular helper functions that expose clearly documented responsibilities;
* optional parallel processing (process pool) for batch workloads;
* defensive error handling and informative logging hooks;
* vectorised/statistical helpers for computing persistence summaries.

Typical usage
-------------

>>> from pathlib import Path
>>> from new_topological_features import (
...     TopologicalConfig,
...     ResidueDescriptor,
...     compute_features_for_residues,
... )
>>> config = TopologicalConfig(
...     neighbor_distance=6.0,
...     filtration_cutoff=8.0,
...     min_persistence=0.01,
...     element_filters=[["C", "N", "O"], "all"],
...     max_alpha_dimension=2,
... )
>>> pdb_path = Path("protein.pdb")
>>> residues = [ResidueDescriptor.from_string("c<A>r<152>R<LEU>")]
>>> df = compute_features_for_residues(pdb_path, residues, config)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gudhi
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

try:  # pragma: no cover
    from .pdb_utils import create_pdb_parser
except ImportError:  # pragma: no cover
    from pdb_utils import create_pdb_parser  # type: ignore

LOGGER = logging.getLogger(__name__)
TRACE_ENV_FLAG = "QTOPO_TOPO_TRACE"
TRACE_ENV_FILTER = "QTOPO_TOPO_TRACE_FILTER"
TRACE_ENV_DIR = "QTOPO_TOPO_TRACE_DIR"


# --------------------------------------------------------------------------- #
# Data classes and configuration                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResidueDescriptor:
    """Canonical representation of a residue identifier.

    The legacy code encoded residues as ``c<A>r<152>R<LEU>`` (optionally with an
    insertion code ``i< >``).  This class parses and stores those components,
    making downstream code easier to test.
    """

    chain_id: str
    residue_number: int
    residue_name: Optional[str] = None
    insertion_code: str = ""
    raw_descriptor: Optional[str] = None

    _DESCRIPTOR_REGEX = re.compile(
        r"c<(?P<chain>[^>]+)>" r"r<(?P<resnum>[-+]?\d+)>" r"(?:i<(?P<icode>[^>]*)>)?" r"(?:R<(?P<resname>[^>]+)>)?"
    )

    @classmethod
    def from_string(cls, descriptor: str) -> "ResidueDescriptor":
        match = cls._DESCRIPTOR_REGEX.fullmatch(descriptor.strip())
        if not match:
            raise ValueError(f"Invalid residue descriptor: {descriptor}")
        return cls(
            chain_id=match.group("chain"),
            residue_number=int(match.group("resnum")),
            residue_name=match.group("resname"),
            insertion_code=(match.group("icode") or "").strip() or " ",
            raw_descriptor=descriptor.strip(),
        )

    @property
    def bio_id(self) -> Tuple[str, int, str]:
        """Return the tuple accepted by Biopython residues."""

        return (" ", self.residue_number, self.insertion_code)

    def to_string(self) -> str:
        if self.raw_descriptor:
            return self.raw_descriptor
        base = f"c<{self.chain_id}>r<{self.residue_number}>"
        if self.insertion_code and self.insertion_code != " ":
            base += f"i<{self.insertion_code}>"
        if self.residue_name:
            base += f"R<{self.residue_name}>"
        return base

@dataclass(frozen=True)
class _TraceConfig:
    enabled: bool
    trace_dir: Path
    filters: Tuple[str, ...]

    @classmethod
    def from_env(cls, pdb_path: Path) -> "_TraceConfig":
        if os.environ.get(TRACE_ENV_FLAG, "").strip() not in {"1", "true", "TRUE", "yes", "Yes"}:
            return cls(enabled=False, trace_dir=Path("."), filters=())
        raw_dir = os.environ.get(TRACE_ENV_DIR)
        if raw_dir:
            trace_dir = Path(raw_dir).expanduser()
        else:
            trace_dir = Path(".") / "topology_trace"
        raw_filter = os.environ.get(TRACE_ENV_FILTER, "")
        filters = tuple(token.strip() for token in raw_filter.split(",") if token.strip())
        return cls(enabled=True, trace_dir=trace_dir, filters=filters)


def _hash_array(arr: np.ndarray) -> Optional[str]:
    if arr is None:
        return None
    try:
        view = np.asarray(arr)
        return hashlib.sha256(view.tobytes()).hexdigest()
    except Exception:
        return None


def _array_stats(arr: np.ndarray) -> Dict[str, object]:
    view = np.asarray(arr)
    if view.size == 0:
        return {"shape": list(view.shape), "size": 0}
    return {
        "shape": list(view.shape),
        "size": int(view.size),
        "min": float(np.min(view)),
        "max": float(np.max(view)),
        "mean": float(np.mean(view)),
        "std": float(np.std(view)),
    }


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


class TopologyTracer:
    """Opt-in tracing for topology drift debugging (no-op when disabled)."""

    def __init__(self, config: _TraceConfig, pdb_path: Path):
        self.config = config
        self.pdb_path = Path(pdb_path)
        self.pdb_token = _safe_label(self.pdb_path.stem)

    @classmethod
    def from_env(cls, pdb_path: Path) -> "TopologyTracer":
        return cls(_TraceConfig.from_env(pdb_path), pdb_path)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _matches_filter(self, residue_label: str) -> bool:
        if not self.config.filters:
            return True
        target = residue_label + " " + self.pdb_token
        return any(token in target for token in self.config.filters)

    def record(self, residue_label: str, stage: str, payload: Dict[str, object]) -> None:
        if not self.enabled or not self._matches_filter(residue_label):
            return
        try:
            root = self.config.trace_dir / self.pdb_token / _safe_label(residue_label)
            root.mkdir(parents=True, exist_ok=True)
            output_path = root / f"{stage}.json"
            serializable = dict(payload)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(serializable, handle, indent=2)
        except Exception:
            LOGGER.exception("Topology tracing failed for %s stage %s", residue_label, stage)

@dataclass(frozen=True)
class ElementFilter:
    label: str
    tokens: Optional[Tuple[str, ...]]  # None means "all"


def _normalise_element_filters(
    filters: Sequence[Sequence[str] | str],
) -> Tuple[ElementFilter, ...]:
    normalised: List[ElementFilter] = []
    for entry in filters:
        if isinstance(entry, str):
            cleaned = entry.strip()
            if cleaned.lower() == "all":
                normalised.append(ElementFilter(label="all", tokens=None))
            else:
                tokens = tuple(ch.strip().upper()[0] for ch in cleaned if ch.strip())
                if not tokens:
                    raise ValueError("Element filters must contain at least one symbol")
                normalised.append(ElementFilter(label=cleaned, tokens=tokens))
        else:
            tokens = tuple(str(ch).strip().upper()[0] for ch in entry if str(ch).strip())
            if not tokens:
                raise ValueError("Element filter sequences must contain at least one symbol")
            label = "".join(tokens)
            normalised.append(ElementFilter(label=label, tokens=tokens))
    return tuple(normalised)


@dataclass(frozen=True)
class TopologicalConfig:
    """Configuration options for topological feature extraction."""

    neighbor_distance: float = 6.0
    filtration_cutoff: float = 8.0
    min_persistence: float = 0.01
    element_filters: Sequence[Sequence[str] | str] = field(default_factory=lambda: ["all"])
    max_rips_dimension: int = 1
    max_alpha_dimension: int = 2
    workers: Optional[int] = None  # None = sequential, otherwise max processes
    log_progress: bool = False
    dedup_sort: bool = False

    def __post_init__(self) -> None:
        if self.neighbor_distance <= 0:
            raise ValueError("neighbor_distance must be positive")
        if self.filtration_cutoff <= 0:
            raise ValueError("filtration_cutoff must be positive")
        if self.min_persistence < 0:
            raise ValueError("min_persistence cannot be negative")
        if self.max_rips_dimension < 1:
            raise ValueError("max_rips_dimension must be >= 1")
        if self.max_alpha_dimension < 1:
            raise ValueError("max_alpha_dimension must be >= 1")
        if self.workers is not None and self.workers <= 0:
            raise ValueError("workers must be positive or None")
        object.__setattr__(self, "element_filters", _normalise_element_filters(self.element_filters))
        object.__setattr__(self, "dedup_sort", bool(self.dedup_sort))


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=16)
def _load_structure_cached(path_str: str) -> Structure:
    parser = create_pdb_parser()
    return parser.get_structure("structure", path_str)


def _load_structure(pdb_path: Path) -> Structure:
    path_str = str(pdb_path.resolve())
    try:
        return _load_structure_cached(path_str)
    except Exception as exc:  # pragma: no cover - error propagated to caller
        raise RuntimeError(f"Failed to parse PDB file {pdb_path}: {exc}") from exc


def _resolve_residue(structure: Structure, descriptor: ResidueDescriptor) -> Residue:
    try:
        residue = structure[0][descriptor.chain_id][descriptor.bio_id]
    except KeyError as exc:
        raise KeyError(f"Residue not found: {descriptor.to_string()}") from exc
    return residue


def _build_neighbor_search(structure: Structure) -> NeighborSearch:
    atoms = list(structure.get_atoms())
    return NeighborSearch(atoms)


def _select_reference_atom(residue: Residue) -> Atom:
    try:
        return residue["CA"]
    except KeyError:
        atoms = list(residue.get_atoms())
        if not atoms:
            raise ValueError(f"Residue {residue} contains no atoms")
        return atoms[0]


def _neighbors(residue_atom: Atom, search: NeighborSearch, radius: float) -> List[Atom]:
    return search.search(residue_atom.coord, radius)


def _filter_atom_coordinates_with_chains(
    atoms: Iterable[Atom],
    element_filter: ElementFilter,
    hbond_residues: Optional[Set[Tuple[str, int, str]]] = None,
    ref_chain: Optional[str] = None,
    hbond_inter_only: bool = False,
) -> Tuple[np.ndarray, List[str], List[bool]]:
    coords: List[np.ndarray] = []
    chains: List[str] = []
    flags: List[bool] = []
    def _is_hbonded(atom: Atom) -> bool:
        if not hbond_residues:
            return False
        res = atom.get_parent()
        chain_id = res.get_parent().id  # type: ignore[attr-defined]
        res_id = res.get_id()
        resnum = int(res_id[1])
        icode = res_id[2] if len(res_id) > 2 else " "
        if hbond_inter_only and ref_chain is not None and chain_id == ref_chain:
            return False
        return (chain_id, resnum, icode) in hbond_residues

    if element_filter.tokens is None:
        for atom in atoms:
            coords.append(atom.coord)
            chain_id = atom.get_parent().get_parent().id  # type: ignore[attr-defined]
            chains.append(chain_id)
            flags.append(_is_hbonded(atom))
    else:
        prefix_set = set(element_filter.tokens)
        for atom in atoms:
            atom_id = atom.get_id()
            if atom_id and atom_id[0].upper() in prefix_set:
                coords.append(atom.coord)
                chain_id = atom.get_parent().get_parent().id  # type: ignore[attr-defined]
                chains.append(chain_id)
                flags.append(_is_hbonded(atom))
    return (np.array(coords, dtype=float) if coords else np.empty((0, 3), dtype=float), chains, flags)


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
        if dim == 0 and death <= cutoff and (death - birth) >= min_persistence
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
        if dim == 1 and (death - birth) >= min_persistence
    ]
    if filtered:
        births = np.array([birth for birth, _ in filtered], dtype=float)
        deaths = np.array([death for _, death in filtered], dtype=float)
        lengths = deaths - births
        stats[:] = [
            float(lengths.sum()),
            float(lengths.min()),
            float(lengths.max()),
            float(lengths.mean()),
            float(lengths.std(ddof=0)),
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
        ]
    return stats


def _compute_features_for_atoms(
    coord_set: np.ndarray,
    *,
    chains: List[str] | None = None,
    bias_mode: Optional[str] = None,
    bias_value: Optional[float] = None,
    config: TopologicalConfig,
    logger: Optional[logging.Logger] = None,
    hbond_flags: Optional[List[bool]] = None,
    hbond_factor: Optional[float] = None,
    hbond_inter_only: bool = False,
    tracer: Optional[TopologyTracer] = None,
    residue_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    log = logger or LOGGER
    coords = np.asarray(coord_set, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinate arrays must have shape (N, 3)")

    # Enforce deterministic ordering to avoid platform-dependent neighbor order
    if coords.size > 0 and chains:
        order = sorted(range(len(coords)), key=lambda idx: (chains[idx], coords[idx, 0], coords[idx, 1], coords[idx, 2]))
        coords = coords[order]
        chains = [chains[idx] for idx in order]
        if hbond_flags is not None:
            hbond_flags = [hbond_flags[idx] for idx in order]

    if coords.size == 0:
        log.debug("No neighbor coordinates available; returning zero features")
        if tracer and tracer.enabled and residue_label:
            tracer.record(
                residue_label,
                "neighbors",
                {"coords_hash": _hash_array(coords), "coords_stats": _array_stats(coords)},
            )
        return np.zeros(5), np.zeros(15)

    # Remove duplicate coordinates (if any)
    if config.dedup_sort and coords.shape[0] > 1:
        unique_coords, indices = np.unique(coords, axis=0, return_index=True)
        if unique_coords.shape[0] != coords.shape[0]:
            log.debug("Removed %d duplicate coordinate(s)", coords.shape[0] - unique_coords.shape[0])
        order = np.argsort(indices)
        coords = unique_coords[order]
        if chains is not None:
            chains = [chains[idx] for idx in order]
        if hbond_flags is not None:
            hbond_flags = [hbond_flags[idx] for idx in order]

    if coords.shape[0] < 2:
        log.debug("Insufficient unique points (%d) for topological features", coords.shape[0])
        return np.zeros(5), np.zeros(15)

    try:
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        penalty = float(bias_value) if (bias_mode == "intra_penalty" and bias_value) else 0.0
        if bias_mode == "intra_penalty" and chains and bias_value:
            n = coords.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if chains[i] == chains[j]:
                        dist_matrix[i, j] += penalty
                        dist_matrix[j, i] += penalty
        if hbond_flags and hbond_factor and hbond_factor > 0:
            n = dist_matrix.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if hbond_flags[i] or hbond_flags[j]:
                        if not hbond_inter_only or (chains and chains[i] != chains[j]):
                            dist_matrix[i, j] *= hbond_factor
                            dist_matrix[j, i] *= hbond_factor
        if penalty > 0:
            rips_complex = gudhi.RipsComplex(
                distance_matrix=dist_matrix.tolist(),
                max_edge_length=config.filtration_cutoff + penalty,
            )
        else:
            rips_complex = gudhi.RipsComplex(distance_matrix=dist_matrix.tolist())
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=config.max_rips_dimension)
        rips_persistence = simplex_tree.persistence()
    except Exception as exc:
        raise RuntimeError(f"Failed to construct Rips complex: {exc}") from exc

    alpha_persistence: Sequence[Tuple[int, Tuple[float, float]]] = []
    if coords.shape[0] >= 3 and config.max_alpha_dimension >= 1:
        try:
            alpha_complex = gudhi.AlphaComplex(points=coords)
            alpha_tree = alpha_complex.create_simplex_tree()
            alpha_persistence = alpha_tree.persistence()
        except Exception as exc:
            raise RuntimeError(f"Failed to construct Alpha complex: {exc}") from exc
    else:
        log.debug(
            "Skipping alpha complex computation (points=%d, max_alpha_dimension=%d)",
            coords.shape[0],
            config.max_alpha_dimension,
        )

    if tracer and tracer.enabled and residue_label:
        tracer.record(
            residue_label,
            "distance_matrix",
            {"hash": _hash_array(dist_matrix), "stats": _array_stats(dist_matrix)},
        )
        tracer.record(
            residue_label,
            "persistence",
            {
                "rips_hash": _hash_array(np.asarray(rips_persistence, dtype=object)),
                "alpha_hash": _hash_array(np.asarray(alpha_persistence, dtype=object)),
                "rips_len": len(rips_persistence),
                "alpha_len": len(alpha_persistence),
            },
        )

    feature_0d = _persistence_summary_zero_dim(
        rips_persistence,
        cutoff=config.filtration_cutoff,
        min_persistence=config.min_persistence,
    )
    feature_1d = _persistence_summary_one_dim(
        alpha_persistence,
        min_persistence=config.min_persistence,
    )
    if tracer and tracer.enabled and residue_label:
        tracer.record(
            residue_label,
            "features",
            {
                "f0_hash": _hash_array(feature_0d),
                "f1_hash": _hash_array(feature_1d),
                "f0_stats": _array_stats(feature_0d),
                "f1_stats": _array_stats(feature_1d),
            },
        )
    return feature_0d, feature_1d


def _feature_column_labels(config: TopologicalConfig) -> Tuple[List[str], List[str]]:
    element_labels = [element.label for element in config.element_filters]

    zero_dim_stats = ["death_sum", "death_min", "death_max", "death_mean", "death_std"]
    zero_dim_cols = [f"f0_{label}_{stat}" for label in element_labels for stat in zero_dim_stats]

    one_dim_stats = [
        "len_sum",
        "len_min",
        "len_max",
        "len_mean",
        "len_std",
        "birth_sum",
        "birth_min",
        "birth_max",
        "birth_mean",
        "birth_std",
        "death_sum",
        "death_min",
        "death_max",
        "death_mean",
        "death_std",
    ]
    one_dim_cols = [f"f1_{label}_{stat}" for label in element_labels for stat in one_dim_stats]

    return zero_dim_cols, one_dim_cols


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def compute_features_for_residue(
    pdb_path: Path,
    residue: ResidueDescriptor,
    config: TopologicalConfig,
    *,
    logger: Optional[logging.Logger] = None,
    structure: Optional[Structure] = None,
    neighbor_search: Optional[NeighborSearch] = None,
    bias_mode: Optional[str] = None,
    bias_value: Optional[float] = None,
    chain_filter: Optional[str] = None,
    sse_mode: bool = False,
    polar_mode: bool = False,
    hbond_weight: bool = False,
    hbond_residues: Optional[Set[Tuple[str, int, str]]] = None,
    hbond_factor: Optional[float] = None,
    hbond_inter_only: bool = False,
    tracer: Optional[TopologyTracer] = None,
) -> List[float]:
    log = logger or LOGGER

    structure_obj = structure or _load_structure(pdb_path)
    search = neighbor_search or _build_neighbor_search(structure_obj)

    residue_obj = _resolve_residue(structure_obj, residue)
    reference_atom = _select_reference_atom(residue_obj)
    neighbors = _neighbors(reference_atom, search, config.neighbor_distance)
    if chain_filter in {"cross_only", "within_chain"}:
        ref_chain = residue.chain_id
        if chain_filter == "cross_only":
            neighbors = [atom for atom in neighbors if atom.get_parent().get_parent().id != ref_chain]  # type: ignore[attr-defined]
        else:
            neighbors = [atom for atom in neighbors if atom.get_parent().get_parent().id == ref_chain]  # type: ignore[attr-defined]

    residue_label = residue.to_string()
    if tracer and tracer.enabled:
        neighbor_descriptors: List[str] = []
        neighbor_coords: List[List[float]] = []
        for atom in neighbors:
            res = atom.get_parent()
            chain_id = res.get_parent().id  # type: ignore[attr-defined]
            res_id = res.get_id()
            resnum = int(res_id[1])
            icode = res_id[2] if len(res_id) > 2 else " "
            neighbor_descriptors.append(f"{chain_id}:{resnum}:{icode}:{atom.get_id()}")
            coord = atom.coord
            neighbor_coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
        tracer.record(
            residue_label,
            "neighbors",
            {
                "count": len(neighbor_descriptors),
                "ids_hash": _hash_array(np.asarray(neighbor_descriptors, dtype=object)),
                "coords_hash": _hash_array(np.asarray(neighbor_coords, dtype=float)),
                "coords_stats": _array_stats(np.asarray(neighbor_coords, dtype=float)),
            },
        )

    feature_0d: List[float] = []
    feature_1d: List[float] = []

    for element_filter in config.element_filters:
        coords, chains, hbond_flags = _filter_atom_coordinates_with_chains(
            neighbors,
            element_filter,
            hbond_residues=hbond_residues if (polar_mode and hbond_weight) else None,
            ref_chain=residue.chain_id,
            hbond_inter_only=hbond_inter_only,
        )
        try:
            zero_dim, one_dim = _compute_features_for_atoms(
                coords,
                chains=chains,
                bias_mode=bias_mode,
                bias_value=bias_value,
                config=config,
                logger=log if config.log_progress else None,
                hbond_flags=hbond_flags if (polar_mode and hbond_weight) else None,
                hbond_factor=hbond_factor if (polar_mode and hbond_weight) else None,
                hbond_inter_only=hbond_inter_only,
                tracer=tracer,
                residue_label=residue_label,
            )
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise RuntimeError(
                f"Topological computation failed for {residue.to_string()} with filter {element_filter.label}: {exc}"
            ) from exc
        feature_0d.extend(zero_dim.tolist())
        feature_1d.extend(one_dim.tolist())

    return feature_0d + feature_1d


def compute_features_for_residues(
    pdb_path: Path,
    residues: Sequence[ResidueDescriptor],
    config: TopologicalConfig,
    *,
    logger: Optional[logging.Logger] = None,
    bias_mode: Optional[str] = None,
    bias_value: Optional[float] = None,
    chain_filter: Optional[str] = None,
    sse_mode: bool = False,
    polar_mode: bool = False,
    hbond_weight: bool = False,
    hbond_residues: Optional[Set[Tuple[str, int, str]]] = None,
    hbond_factor: Optional[float] = None,
    hbond_inter_only: bool = False,
    tracer: Optional[TopologyTracer] = None,
) -> pd.DataFrame:
    log = logger or LOGGER

    if not residues:
        return pd.DataFrame()

    worker_count = config.workers if config.workers and config.workers > 1 else 1
    log.debug(
        "Topological feature extraction for %d residue(s) with %d worker(s)",
        len(residues),
        worker_count,
    )

    trace_cfg = tracer or TopologyTracer.from_env(pdb_path)
    active_tracer = trace_cfg if trace_cfg.enabled and worker_count == 1 else None
    if trace_cfg.enabled and worker_count > 1:
        log.warning("Topology tracing enabled but workers > 1; tracing disabled for this run.")

    feature_map: Dict[int, List[float]] = {}
    id_map: Dict[int, str] = {}

    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    _compute_features_worker,
                    (
                        index,
                        pdb_path,
                        descriptor,
                        config,
                        bias_mode,
                        bias_value,
                        chain_filter,
                        sse_mode,
                        polar_mode,
                        hbond_weight,
                        hbond_residues,
                        hbond_factor,
                        hbond_inter_only,
                    ),
                ): index
                for index, descriptor in enumerate(residues)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                descriptor = residues[index]
                try:
                    idx, desc_str, feature_values = future.result()
                except Exception as exc:  # pragma: no cover - propagated to caller
                    raise RuntimeError(
                        f"Failed to compute features for {descriptor.to_string()}: {exc}"
                    ) from exc
                feature_map[idx] = feature_values
                id_map[idx] = desc_str
                if config.log_progress:
                    log.info("Finished %s", desc_str)
    else:
        structure = _load_structure(pdb_path)
        neighbor_search = _build_neighbor_search(structure)
        for index, descriptor in enumerate(residues):
            desc_str = descriptor.to_string()
            if config.log_progress:
                log.info("Processing %s", desc_str)
            feature_values = compute_features_for_residue(
                pdb_path,
                descriptor,
                config,
                logger=log if config.log_progress else None,
                structure=structure,
                neighbor_search=neighbor_search,
                bias_mode=bias_mode,
                bias_value=bias_value,
                chain_filter=chain_filter,
                sse_mode=sse_mode,
                polar_mode=polar_mode,
                hbond_weight=hbond_weight,
                hbond_residues=hbond_residues,
                hbond_factor=hbond_factor,
                hbond_inter_only=hbond_inter_only,
                tracer=active_tracer,
            )
            feature_map[index] = feature_values
            id_map[index] = desc_str

    zero_cols, one_cols = _feature_column_labels(config)
    ordered_rows = [feature_map[idx] for idx in range(len(residues))]
    frame = pd.DataFrame(ordered_rows, columns=zero_cols + one_cols)
    frame.insert(0, "ID", [id_map[idx] for idx in range(len(residues))])
    return frame


# --------------------------------------------------------------------------- #
# Worker helper for process pools                                             #
# --------------------------------------------------------------------------- #


def _compute_features_worker(
    task: Tuple[int, Path, ResidueDescriptor, TopologicalConfig, Optional[str], Optional[float], Optional[str], bool, bool, bool, Optional[Set[Tuple[str, int, str]]], Optional[float], bool]
) -> Tuple[int, str, List[float]]:
    (
        index,
        pdb_path,
        descriptor,
        config,
        bias_mode,
        bias_value,
        chain_filter,
        sse_mode,
        polar_mode,
        hbond_weight,
        hbond_residues,
        hbond_factor,
        hbond_inter_only,
    ) = task
    features = compute_features_for_residue(
        pdb_path,
        descriptor,
        config,
        bias_mode=bias_mode,
        bias_value=bias_value,
        chain_filter=chain_filter,
        sse_mode=sse_mode,
        polar_mode=polar_mode,
        hbond_weight=hbond_weight,
        hbond_residues=hbond_residues,
        hbond_factor=hbond_factor,
        hbond_inter_only=hbond_inter_only,
    )
    return index, descriptor.to_string(), features


__all__ = [
    "ResidueDescriptor",
    "ElementFilter",
    "TopologicalConfig",
    "TopologyTracer",
    "compute_features_for_residue",
    "compute_features_for_residues",
]
