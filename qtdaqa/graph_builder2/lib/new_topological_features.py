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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gudhi
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

LOGGER = logging.getLogger(__name__)


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


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=16)
def _load_structure_cached(path_str: str) -> Structure:
    parser = PDBParser(QUIET=True)
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


def _filter_atom_coordinates(atoms: Iterable[Atom], element_filter: ElementFilter) -> np.ndarray:
    if element_filter.tokens is None:
        coords = [atom.coord for atom in atoms]
    else:
        prefix_set = set(element_filter.tokens)
        coords = [atom.coord for atom in atoms if atom.get_id() and atom.get_id()[0].upper() in prefix_set]
    return np.array(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


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
    config: TopologicalConfig,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    log = logger or LOGGER
    coords = np.asarray(coord_set, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinate arrays must have shape (N, 3)")

    if coords.size == 0:
        log.debug("No neighbor coordinates available; returning zero features")
        return np.zeros(5), np.zeros(15)

    # Remove duplicate coordinates (if any)
    if coords.shape[0] > 1:
        unique_coords = np.unique(coords, axis=0)
        if unique_coords.shape[0] != coords.shape[0]:
            log.debug("Removed %d duplicate coordinate(s)", coords.shape[0] - unique_coords.shape[0])
        coords = unique_coords

    if coords.shape[0] < 2:
        log.debug("Insufficient unique points (%d) for topological features", coords.shape[0])
        return np.zeros(5), np.zeros(15)

    try:
        rips_complex = gudhi.RipsComplex(points=coords)
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

    feature_0d = _persistence_summary_zero_dim(
        rips_persistence,
        cutoff=config.filtration_cutoff,
        min_persistence=config.min_persistence,
    )
    feature_1d = _persistence_summary_one_dim(
        alpha_persistence,
        min_persistence=config.min_persistence,
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
) -> List[float]:
    log = logger or LOGGER

    structure_obj = structure or _load_structure(pdb_path)
    search = neighbor_search or _build_neighbor_search(structure_obj)

    residue_obj = _resolve_residue(structure_obj, residue)
    reference_atom = _select_reference_atom(residue_obj)
    neighbors = _neighbors(reference_atom, search, config.neighbor_distance)

    feature_0d: List[float] = []
    feature_1d: List[float] = []

    for element_filter in config.element_filters:
        coords = _filter_atom_coordinates(neighbors, element_filter)
        try:
            zero_dim, one_dim = _compute_features_for_atoms(
                coords,
                config=config,
                logger=log if config.log_progress else None,
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

    feature_map: Dict[int, List[float]] = {}
    id_map: Dict[int, str] = {}

    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(_compute_features_worker, (index, pdb_path, descriptor, config)): index
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
    task: Tuple[int, Path, ResidueDescriptor, TopologicalConfig]
) -> Tuple[int, str, List[float]]:
    index, pdb_path, descriptor, config = task
    features = compute_features_for_residue(pdb_path, descriptor, config)
    return index, descriptor.to_string(), features


__all__ = [
    "ResidueDescriptor",
    "ElementFilter",
    "TopologicalConfig",
    "compute_features_for_residue",
    "compute_features_for_residues",
]
