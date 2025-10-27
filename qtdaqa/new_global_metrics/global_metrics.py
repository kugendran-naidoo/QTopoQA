#!/usr/bin/env python3
"""Compute global metrics for interface-focused training (initially: contact count)."""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import math
import numpy as np
from Bio.PDB import NeighborSearch, PDBParser, PDBIO, ShrakeRupley
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from log_dirs import LogDirectoryInfo, prepare_log_directory

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_CONTACT_CUTOFF = 5.0  # Ångström
DEFAULT_SASA_PROBE_RADIUS = 1.4
DEFAULT_SASA_SPHERE_POINTS = 100
DEFAULT_SOFTMIN_ALPHA = 2.0
DEFAULT_GYRATION_NORMALISER = 1.0
_SASA_FALLBACK_WARNED = False
EPSILON = 1e-6


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _configure_logger(run_log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("global_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(run_log_dir / "global_metrics.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def _parse_common_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing decoy PDBs")
    parser.add_argument("--work-dir", type=Path, required=True, help="Working directory for intermediates and outputs")
    parser.add_argument("--graph-dir", type=Path, required=True, help="Placeholder for graph outputs (kept for CLI parity)")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Root directory for logs")
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes (future use)")
    parser.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF, help="Heavy-atom contact cutoff (Å)")
    parser.add_argument("--output-csv", type=Path, default=Path("global_metrics.csv"), help="Relative or absolute CSV output path")


class MetricFeature:
    """Base class for metric features that can be toggled via CLI."""

    name: str
    cli_name: str
    columns: Sequence[str]
    dependencies: Sequence[str] = ()

    def __init__(self) -> None:
        dest = self.cli_name.replace("-", "_")
        self._disabled_dest = f"no_{dest}"

    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"--no-{self.cli_name}",
            action="store_true",
            dest=self._disabled_dest,
            help=f"Skip generation of {self.name.replace('_', ' ')} feature",
        )

    def should_run(self, args: argparse.Namespace, enabled: Set[str]) -> bool:
        if getattr(args, self._disabled_dest, False):
            return False
        return all(dep in enabled for dep in self.dependencies)

    def compute(self, context: "ModelContext") -> Mapping[str, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    residue_seq: int
    insertion_code: str
    residue_name: str

    def sort_key(self) -> Tuple[str, int, str, str]:
        return (self.chain_id, self.residue_seq, self.insertion_code, self.residue_name)


@dataclass(frozen=True)
class AtomKey:
    residue: ResidueKey
    atom_name: str

    def sort_key(self) -> Tuple[str, int, str, str, str]:
        res_key = self.residue.sort_key()
        return (*res_key, self.atom_name)


@dataclass(frozen=True)
class ContactSummary:
    atom_pairs: Tuple[Tuple[AtomKey, AtomKey], ...]
    residue_pairs: Tuple[Tuple[ResidueKey, ResidueKey], ...]
    residues: Tuple[ResidueKey, ...]

    @property
    def atom_contact_count(self) -> int:
        return len(self.atom_pairs)


@dataclass(frozen=True)
class ChainPairCentroidData:
    chain_a: str
    chain_b: str
    distance: float
    normalized_distance: float
    weight: float


class ContactAccumulator:
    def __init__(self) -> None:
        self._atom_pairs: Set[Tuple[AtomKey, AtomKey]] = set()
        self._residue_pairs: Set[Tuple[ResidueKey, ResidueKey]] = set()
        self._residues: Set[ResidueKey] = set()

    def record(self, atom_a: Atom, atom_b: Atom) -> None:
        key_a = _atom_key(atom_a)
        key_b = _atom_key(atom_b)
        if key_a is None or key_b is None:
            return
        ordered = _order_atom_pair(key_a, key_b)
        self._atom_pairs.add(ordered)
        self._residue_pairs.add((key_a.residue, key_b.residue))
        self._residues.add(key_a.residue)
        self._residues.add(key_b.residue)

    def build(self) -> ContactSummary:
        return ContactSummary(
            atom_pairs=tuple(sorted(self._atom_pairs, key=lambda pair: (pair[0].sort_key(), pair[1].sort_key()))),
            residue_pairs=tuple(sorted(self._residue_pairs, key=lambda pair: (pair[0].sort_key(), pair[1].sort_key()))),
            residues=tuple(sorted(self._residues, key=lambda res: res.sort_key())),
        )


def _atom_key(atom: Atom) -> Optional[AtomKey]:
    residue: Residue = atom.get_parent()
    chain: Chain = residue.get_parent()

    element = (atom.element or "").strip().upper()
    if element == "H":
        return None

    res_id = residue.id  # (hetero flag, sequence number, insertion code)
    res_seq = int(res_id[1])
    insertion = (res_id[2] or "").strip()
    residue_key = ResidueKey(
        chain_id=str(getattr(chain, "id", "")),
        residue_seq=res_seq,
        insertion_code=insertion,
        residue_name=residue.get_resname().strip(),
    )
    return AtomKey(residue=residue_key, atom_name=atom.get_name().strip())


def _order_atom_pair(atom_a: AtomKey, atom_b: AtomKey) -> Tuple[AtomKey, AtomKey]:
    if atom_a.sort_key() <= atom_b.sort_key():
        return atom_a, atom_b
    return atom_b, atom_a


def _iter_heavy_atoms(chain: Chain) -> Iterator[Atom]:
    for atom in chain.get_atoms():
        element = (atom.element or "").strip().upper()
        if element != "H":
            yield atom


def _select_primary_model(structure: Structure) -> Optional[Model]:
    try:
        return next(structure.get_models())
    except StopIteration:
        return None


def _iter_chain_pairs(model: Model) -> Iterator[Tuple[Chain, Chain]]:
    chains = [chain for chain in model]
    for idx, chain_a in enumerate(chains):
        for chain_b in chains[idx + 1 :]:
            yield chain_a, chain_b


def _build_contact_summary(structure: Structure, cutoff: float) -> ContactSummary:
    model = _select_primary_model(structure)
    if model is None:
        return ContactSummary(atom_pairs=tuple(), residue_pairs=tuple(), residues=tuple())

    accumulator = ContactAccumulator()
    for chain_a, chain_b in _iter_chain_pairs(model):
        atoms_b = list(_iter_heavy_atoms(chain_b))
        if not atoms_b:
            continue
        neighbor_search = NeighborSearch(atoms_b)
        for atom_a in _iter_heavy_atoms(chain_a):
            for atom_b in neighbor_search.search(atom_a.coord, cutoff):
                accumulator.record(atom_a, atom_b)
    return accumulator.build()


def _compute_buried_sasa(context: ModelContext) -> float:
    try:
        import MDAnalysis as mda  # type: ignore
        from MDAnalysis.analysis.sasa import ShrakeRupley as MDAShrakeRupley  # type: ignore
    except ImportError:
        return _compute_buried_sasa_fallback(context, warn=True)

    universe = mda.Universe(str(context.pdb_path))
    if not len(universe.atoms):
        return 0.0

    def _shrake_rupley_area(atom_group) -> float:
        calculator = MDAShrakeRupley(
            atom_group,
            probe_radius=DEFAULT_SASA_PROBE_RADIUS,
            n_sphere_points=DEFAULT_SASA_SPHERE_POINTS,
        )
        calculator.run()
        return float(calculator.results.total_area)

    complex_area = _shrake_rupley_area(universe.atoms)

    chain_groups: List = []
    chain_groups_map: Dict[str, object] = {}

    for chain_id, atom_group in universe.atoms.groupby("chainIDs").items():
        if not len(atom_group):
            continue
        key = chain_id.strip() or f"chain_{len(chain_groups_map)}"
        chain_groups_map[key] = atom_group

    if len(chain_groups_map) <= 1:
        for segment in universe.segments:
            if len(segment.atoms):
                key = segment.segid.strip() or f"segment_{len(chain_groups_map)}"
                chain_groups_map.setdefault(key, segment.atoms)

    chain_groups = list(chain_groups_map.values()) or [universe.atoms]

    monomer_area = 0.0
    for atom_group in chain_groups:
        monomer_area += _shrake_rupley_area(atom_group)

    buried = monomer_area - complex_area
    if buried < 0.0:
        buried = 0.0
    return buried


def _compute_buried_sasa_fallback(context: ModelContext, *, warn: bool = False) -> float:
    global _SASA_FALLBACK_WARNED
    if warn and not _SASA_FALLBACK_WARNED:
        logging.getLogger("global_metrics").warning(
            "MDAnalysis Shrake–Rupley unavailable; falling back to FreeSASA/Biopython implementation."
        )
        _SASA_FALLBACK_WARNED = True

    try:
        return _compute_buried_sasa_freesasa(context)
    except Exception as exc:  # pragma: no cover - fallback path
        logging.getLogger("global_metrics").warning(
            "FreeSASA fallback failed for %s: %s; retrying with Biopython Shrake–Rupley.",
            context.pdb_path,
            exc,
        )
        return _compute_buried_sasa_biopython(context)


def _compute_buried_sasa_freesasa(context: ModelContext) -> float:
    import freesasa  # type: ignore

    parameters = freesasa.Parameters()
    parameters.setProbeRadius(float(DEFAULT_SASA_PROBE_RADIUS))
    parameters.setNPoints(int(DEFAULT_SASA_SPHERE_POINTS))

    structure = freesasa.Structure(str(context.pdb_path))
    complex_result = freesasa.calc(structure, parameters)
    complex_area = float(complex_result.totalArea())

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sasa", str(context.pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return 0.0

    monomer_area = 0.0
    for chain in model:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            io = PDBIO()
            io.set_structure(chain)
            io.save(str(tmp_path))
            chain_structure = freesasa.Structure(str(tmp_path))
            chain_result = freesasa.calc(chain_structure, parameters)
            monomer_area += float(chain_result.totalArea())
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    buried = monomer_area - complex_area
    return float(buried if buried > 0.0 else 0.0)


def _compute_buried_sasa_biopython(context: ModelContext) -> float:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sasa", str(context.pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return 0.0

    complex_area = _biopython_sasa_area(model)

    complex_area = _biopython_sasa_area(model)
    monomer_area = 0.0
    for chain in model:
        monomer_area += _biopython_sasa_area(chain)

    buried = monomer_area - complex_area
    return float(buried if buried > 0.0 else 0.0)


def _biopython_sasa_area(entity) -> float:
    calculator = ShrakeRupley(
        probe_radius=DEFAULT_SASA_PROBE_RADIUS,
        n_points=DEFAULT_SASA_SPHERE_POINTS,
    )
    calculator.compute(entity, level="A")
    total = 0.0
    for atom in entity.get_atoms():
        value = getattr(atom, "sasa", None)
        if value is None:
            value = atom.xtra.get("EXP_SASA", 0.0)
        total += float(value or 0.0)
    return total


def _residue_atom_coords(structure: Structure, residue_key: ResidueKey) -> Optional[np.ndarray]:
    model = _select_primary_model(structure)
    if model is None:
        return None
    try:
        chain = model[residue_key.chain_id]
    except KeyError:
        return None
    insertion = residue_key.insertion_code or " "
    try:
        residue = chain[(" ", residue_key.residue_seq, insertion)]
    except KeyError:
        return None

    coords: List[np.ndarray] = []
    for atom in residue.get_atoms():
        element = (atom.element or "").strip().upper()
        if element == "H":
            continue
        coords.append(np.asarray(atom.coord, dtype=float))

    if not coords:
        for atom in residue.get_atoms():
            coords.append(np.asarray(atom.coord, dtype=float))

    if not coords:
        return None
    return np.vstack(coords)


def _chain_interface_coords(structure: Structure, residue_keys: Set[ResidueKey]) -> Optional[np.ndarray]:
    coord_blocks: List[np.ndarray] = []
    for key in residue_keys:
        coords = _residue_atom_coords(structure, key)
        if coords is not None:
            coord_blocks.append(coords)
    if not coord_blocks:
        return None
    return np.vstack(coord_blocks)


def _radius_of_gyration(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 0.0
    centroid = np.mean(coords, axis=0)
    diffs = coords - centroid
    squared = np.sum(diffs * diffs, axis=1)
    return float(np.sqrt(max(EPSILON, np.mean(squared))))


def _collect_chain_pair_centroids(context: ModelContext) -> List[ChainPairCentroidData]:
    summary = context.get_contact_summary()
    pair_map: Dict[Tuple[str, str], Dict[str, object]] = {}

    for atom_a, atom_b in summary.atom_pairs:
        chain_a = atom_a.residue.chain_id
        chain_b = atom_b.residue.chain_id
        if chain_a == chain_b:
            continue
        if chain_a <= chain_b:
            key = (chain_a, chain_b)
            res_a, res_b = atom_a.residue, atom_b.residue
        else:
            key = (chain_b, chain_a)
            res_a, res_b = atom_b.residue, atom_a.residue
        record = pair_map.setdefault(
            key,
            {
                "contacts": 0,
                "residues": {key[0]: set(), key[1]: set()},
            },
        )
        record["contacts"] = int(record["contacts"]) + 1
        residues: Dict[str, Set[ResidueKey]] = record["residues"]  # type: ignore[assignment]
        residues.setdefault(chain_a, set()).add(res_a)
        residues.setdefault(chain_b, set()).add(res_b)

    structure = context.structure
    pair_data: List[ChainPairCentroidData] = []

    for (chain_a, chain_b), record in pair_map.items():
        residues: Dict[str, Set[ResidueKey]] = record["residues"]  # type: ignore[assignment]
        coords_a = _chain_interface_coords(structure, residues.get(chain_a, set()))
        coords_b = _chain_interface_coords(structure, residues.get(chain_b, set()))
        if coords_a is None or coords_b is None:
            continue
        centroid_a = np.mean(coords_a, axis=0)
        centroid_b = np.mean(coords_b, axis=0)
        distance = float(np.linalg.norm(centroid_a - centroid_b))
        rg_a = _radius_of_gyration(coords_a)
        rg_b = _radius_of_gyration(coords_b)
        scale = max(EPSILON, (rg_a + rg_b) / 2.0)
        normalized = distance / scale
        weight = float(max(1, int(record["contacts"])))
        pair_data.append(
            ChainPairCentroidData(
                chain_a=chain_a,
                chain_b=chain_b,
                distance=distance,
                normalized_distance=normalized,
                weight=weight,
            )
        )

    return pair_data


def _softmin(values: Sequence[float], alpha: float = DEFAULT_SOFTMIN_ALPHA) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    min_val = float(np.min(arr))
    shifted = np.exp(-alpha * (arr - min_val))
    mean_shifted = float(np.mean(shifted))
    if mean_shifted <= 0.0:
        return min_val
    return min_val - (math.log(mean_shifted) / alpha)


def _compute_centroid_statistics(context: ModelContext) -> Dict[str, float]:
    pair_data = _collect_chain_pair_centroids(context)
    if not pair_data:
        return {"median": 0.0, "softmin": 0.0, "weighted_mean": 0.0}

    values = [data.normalized_distance for data in pair_data]
    median_value = float(statistics.median(values))
    softmin_value = float(_softmin(values, DEFAULT_SOFTMIN_ALPHA))

    weights = [max(data.weight, EPSILON) for data in pair_data]
    weighted_mean = float(sum(v * w for v, w in zip(values, weights)) / sum(weights))

    return {
        "median": median_value,
        "softmin": softmin_value,
        "weighted_mean": weighted_mean,
    }


def _chain_atom_coords(chain: Chain) -> Optional[np.ndarray]:
    coords: List[np.ndarray] = []
    for residue in chain:
        for atom in residue.get_atoms():
            element = (atom.element or "").strip().upper()
            if element == "H":
                continue
            coords.append(np.asarray(atom.coord, dtype=float))
    if not coords:
        for residue in chain:
            for atom in residue.get_atoms():
                coords.append(np.asarray(atom.coord, dtype=float))
    if not coords:
        return None
    return np.vstack(coords)


def _compute_chain_radius_of_gyration(structure: Structure) -> Dict[str, float]:
    model = _select_primary_model(structure)
    if model is None:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}

    rg_values: List[float] = []
    for chain in model:
        coords = _chain_atom_coords(chain)
        if coords is None:
            continue
        rg = _radius_of_gyration(coords)
        if rg > 0.0:
            rg_values.append(rg)

    if not rg_values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}

    mean_value = float(statistics.mean(rg_values))
    median_value = float(statistics.median(rg_values))
    std_value = float(np.std(np.asarray(rg_values, dtype=float))) if len(rg_values) > 1 else 0.0

    return {"mean": mean_value, "median": median_value, "std": std_value}


def _compute_chain_length_features(structure: Structure) -> Dict[str, float]:
    model = _select_primary_model(structure)
    if model is None:
        return {
            "sequence_largest_fraction": 0.0,
            "sequence_top2_fraction": 0.0,
            "sequence_log_L1_L2": 0.0,
            "sequence_max_min_ratio_clipped": 0.0,
            "sequence_cv": 0.0,
            "sequence_median_length": 0.0,
            "sequence_mean_length": 0.0,
            "sequence_p75_length": 0.0,
            "sequence_n_long_ge_1p5med": 0.0,
            "sequence_n_short_le_0p5med": 0.0,
            "sequence_chain_count": 0.0,
            "sequence_total_length": 0.0,
        }

    lengths: Dict[str, int] = {}
    for chain in model:
        count = 0
        for residue in chain.get_residues():
            hetero = residue.id[0]
            if hetero.strip() not in {"H", "W"}:  # include standard amino acids and most hetero peptide residues
                count += 1
        chain_id = getattr(chain, "id", "") or f"chain_{len(lengths)}"
        if count > 0:
            lengths[chain_id] = count

    if not lengths:
        return {
            "sequence_largest_fraction": 0.0,
            "sequence_top2_fraction": 0.0,
            "sequence_log_L1_L2": 0.0,
            "sequence_max_min_ratio_clipped": 0.0,
            "sequence_cv": 0.0,
            "sequence_median_length": 0.0,
            "sequence_mean_length": 0.0,
            "sequence_p75_length": 0.0,
            "sequence_n_long_ge_1p5med": 0.0,
            "sequence_n_short_le_0p5med": 0.0,
            "sequence_chain_count": 0.0,
            "sequence_total_length": 0.0,
        }

    L = np.array(sorted(lengths.values(), reverse=True), dtype=float)
    k = len(L)
    T = float(L.sum())
    mu = float(L.mean())
    med = float(np.median(L))
    eps = 1e-8
    L1 = float(L[0])
    L2 = float(L[1]) if k > 1 else L1
    Lk = float(L[-1])

    feats = {
        "sequence_largest_fraction": L1 / (T + eps),
        "sequence_top2_fraction": (L1 + (L2 if k > 1 else 0.0)) / (T + eps),
        "sequence_log_L1_L2": float(np.log((L1 + eps) / (L2 + eps))),
        "sequence_max_min_ratio_clipped": float(min(L1 / (Lk + eps), 10.0)),
        "sequence_cv": float(np.std(L) / (mu + eps)),
        "sequence_median_length": med,
        "sequence_mean_length": mu,
        "sequence_p75_length": float(np.quantile(L, 0.75)),
        "sequence_n_long_ge_1p5med": float((L >= 1.5 * med).sum()),
        "sequence_n_short_le_0p5med": float((L <= 0.5 * med).sum()),
        "sequence_chain_count": float(k),
        "sequence_total_length": T,
    }
    return feats


@dataclass
class ModelContext:
    pdb_path: Path
    structure: Structure
    contact_cutoff: float
    _contact_summary: Optional[ContactSummary] = field(default=None, init=False, repr=False)
    _buried_sasa: Optional[float] = field(default=None, init=False, repr=False)
    _centroid_stats: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)
    _chain_rg_stats: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)
    _chain_length_stats: Optional[Dict[str, float]] = field(default=None, init=False, repr=False)

    def get_contact_summary(self) -> ContactSummary:
        if self._contact_summary is None:
            self._contact_summary = _build_contact_summary(self.structure, self.contact_cutoff)
        return self._contact_summary

    def get_buried_sasa(self) -> float:
        if self._buried_sasa is None:
            self._buried_sasa = _compute_buried_sasa(self)
        return self._buried_sasa

    def get_centroid_stats(self) -> Dict[str, float]:
        if self._centroid_stats is None:
            self._centroid_stats = _compute_centroid_statistics(self)
        return self._centroid_stats

    def get_chain_rg_stats(self) -> Dict[str, float]:
        if self._chain_rg_stats is None:
            self._chain_rg_stats = _compute_chain_radius_of_gyration(self.structure)
        return self._chain_rg_stats

    def get_chain_length_stats(self) -> Dict[str, float]:
        if self._chain_length_stats is None:
            self._chain_length_stats = _compute_chain_length_features(self.structure)
        return self._chain_length_stats


@dataclass(frozen=True)
class MetricConfig:
    dataset_dir: Path
    work_dir: Path
    graph_dir: Path
    log_dir: Path
    jobs: int
    contact_cutoff: float
    output_csv: Path


class InterfaceContactCountFeature(MetricFeature):
    name = "interface_contact_count"
    cli_name = "interface-contact-count"
    columns = ("interface_contact_count",)

    def compute(self, context: ModelContext) -> Mapping[str, float]:
        summary = context.get_contact_summary()
        return {"interface_contact_count": float(summary.atom_contact_count)}


class InterfaceResidueCountFeature(MetricFeature):
    name = "interface_residue_count"
    cli_name = "interface-residue-count"
    columns = ("interface_residue_count",)

    def compute(self, context: ModelContext) -> Mapping[str, float]:
        summary = context.get_contact_summary()
        return {"interface_residue_count": float(len(summary.residues))}


class BuriedSasaFeature(MetricFeature):
    name = "interface_buried_sasa"
    cli_name = "buried-sasa"
    columns = ("interface_buried_sasa",)

    def compute(self, context: ModelContext) -> Mapping[str, float]:
        buried = context.get_buried_sasa()
        return {"interface_buried_sasa": float(buried)}


class InterfaceCentroidDistanceFeature(MetricFeature):
    name = "interface_centroid_distance"
    cli_name = "interface-centroid-distance"
    columns = (
        "interface_centroid_distance_median",
        "interface_centroid_distance_softmin",
        "interface_centroid_distance_weighted_mean",
    )

    def compute(self, context: ModelContext) -> Mapping[str, float]:
        stats = context.get_centroid_stats()
        return {
            "interface_centroid_distance_median": stats["median"],
            "interface_centroid_distance_softmin": stats["softmin"],
            "interface_centroid_distance_weighted_mean": stats["weighted_mean"],
        }


class ChainRadiusOfGyrationFeature(MetricFeature):
    name = "chain_radius_of_gyration"
    cli_name = "radius-gyration"
    columns = (
        "chain_radius_of_gyration_mean",
        "chain_radius_of_gyration_median",
        "chain_radius_of_gyration_std",
    )

    def compute(self, context: ModelContext) -> Mapping[str, float]:
        stats = context.get_chain_rg_stats()
        return {
            "chain_radius_of_gyration_mean": stats["mean"],
            "chain_radius_of_gyration_median": stats["median"],
            "chain_radius_of_gyration_std": stats["std"],
        }


class SequenceLengthRatioFeature(MetricFeature):
    name = "sequence_length_ratio"
    cli_name = "sequence-ratio"
    columns = (
        "sequence_largest_fraction",
        "sequence_top2_fraction",
        "sequence_log_L1_L2",
        "sequence_max_min_ratio_clipped",
        "sequence_cv",
        "sequence_median_length",
        "sequence_mean_length",
        "sequence_p75_length",
        "sequence_n_long_ge_1p5med",
        "sequence_n_short_le_0p5med",
        "sequence_chain_count",
        "sequence_total_length",
    )

    def compute(self, context: ModelContext) -> Mapping[str, float]:
        return context.get_chain_length_stats()


FEATURES: Tuple[MetricFeature, ...] = (
    InterfaceContactCountFeature(),
    InterfaceResidueCountFeature(),
    BuriedSasaFeature(),
    InterfaceCentroidDistanceFeature(),
    ChainRadiusOfGyrationFeature(),
    SequenceLengthRatioFeature(),
)


class GlobalMetricsRunner:
    def __init__(self, config: MetricConfig, features: Sequence[MetricFeature], logger: logging.Logger) -> None:
        self.config = config
        self.features = features
        self.logger = logger
        self._parser = PDBParser(QUIET=True)

    def run(self) -> int:
        pdb_files = self._discover_pdbs()
        if not pdb_files:
            self.logger.warning("No PDB files found under %s; nothing to compute", self.config.dataset_dir)
            return 0

        if self.config.jobs > 1:
            self.logger.info("Parallel execution not yet implemented; running sequentially with 1 worker")

        feature_names = {feature.name for feature in self.features}
        if "interface_contact_count" in feature_names:
            self.logger.info("Starting interface_contact_count feature computation")
        if "interface_residue_count" in feature_names:
            self.logger.info("Starting interface_residue_count feature computation")
        if "interface_buried_sasa" in feature_names:
            self.logger.info("Starting interface_buried_sasa feature computation")
        if "interface_centroid_distance" in feature_names:
            self.logger.info("Starting interface_centroid_distance feature computation")
        if "chain_radius_of_gyration" in feature_names:
            self.logger.info("Starting chain_radius_of_gyration feature computation")
        if "sequence_length_ratio" in feature_names:
            self.logger.info("Starting sequence_length_ratio feature computation")

        _ensure_dir(self.config.work_dir)
        rows: List[Dict[str, float]] = []
        stats: Dict[str, List[float]] = {column: [] for feature in self.features for column in feature.columns}
        failures: List[Tuple[Path, str]] = []

        for index, pdb_path in enumerate(pdb_files, start=1):
            model_name = self._model_identifier(pdb_path)
            try:
                structure = self._parser.get_structure(model_name, str(pdb_path))
            except Exception as exc:
                failures.append((pdb_path, f"parse error: {exc}"))
                continue

            context = ModelContext(
                pdb_path=pdb_path,
                structure=structure,
                contact_cutoff=self.config.contact_cutoff,
            )

            row: Dict[str, float] = {"MODEL": model_name}
            for feature in self.features:
                try:
                    values = feature.compute(context)
                except Exception as exc:
                    failures.append((pdb_path, f"{feature.name} failed: {exc}"))
                    break
                for column, value in values.items():
                    row[column] = value
                    stats[column].append(value)
            else:
                rows.append(row)
                if index % 50 == 0:
                    self.logger.info("Processed %d/%d models", index, len(pdb_files))

        if rows:
            self._write_csv(rows)
            self._log_stats(stats)

        if failures:
            for path, reason in failures:
                self.logger.error("Failed to process %s (%s)", path, reason)
            self.logger.warning("%d models failed; see log for details", len(failures))

        self.logger.info("Finished processing %d models (success=%d, failed=%d)", len(pdb_files), len(rows), len(failures))
        return 0 if rows else 2 if failures else 0

    def _discover_pdbs(self) -> List[Path]:
        return sorted(path for path in self.config.dataset_dir.rglob("*.pdb") if path.is_file())

    def _model_identifier(self, pdb_path: Path) -> str:
        try:
            relative = pdb_path.relative_to(self.config.dataset_dir)
        except ValueError:
            relative = pdb_path.name
        model_str = relative.as_posix()
        if model_str.lower().endswith(".pdb"):
            model_str = model_str[: -len(".pdb")]
        return model_str

    def _write_csv(self, rows: Sequence[Mapping[str, float]]) -> None:
        columns = ["MODEL"]
        for feature in self.features:
            for column in feature.columns:
                if column not in columns:
                    columns.append(column)

        output_path = self.config.output_csv
        _ensure_dir(output_path.parent)

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        self.logger.info("Wrote %d rows to %s", len(rows), output_path)

    def _log_stats(self, stats: Mapping[str, Sequence[float]]) -> None:
        for column, values in stats.items():
            if not values:
                continue
            try:
                mean_value = statistics.mean(values)
            except statistics.StatisticsError:
                mean_value = float("nan")
            self.logger.info(
                "%s stats — min=%.0f max=%.0f mean=%.2f",
                column,
                min(values),
                max(values),
                mean_value,
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute standalone global metrics for interface learning.")
    _parse_common_cli(parser)
    for feature in FEATURES:
        feature.add_cli_arguments(parser)
    return parser.parse_args(argv)


def _resolve_output_path(args: argparse.Namespace) -> Path:
    output_csv: Path = args.output_csv
    if output_csv.is_absolute():
        return output_csv
    return args.work_dir / output_csv


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        log_info: LogDirectoryInfo = prepare_log_directory(args.log_dir, run_prefix="global_metrics")
    except Exception as exc:
        print(f"Error preparing log directory: {exc}", file=sys.stderr)
        return 2

    logger = _configure_logger(log_info.run_dir)

    if not args.dataset_dir.exists():
        logger.error("Dataset directory does not exist: %s", args.dataset_dir)
        return 1

    enabled_feature_names: Set[str] = set()
    selected_features: List[MetricFeature] = []
    for feature in FEATURES:
        if feature.should_run(args, enabled_feature_names | {feature.name}):
            selected_features.append(feature)
            enabled_feature_names.add(feature.name)

    if not selected_features:
        logger.warning("All features disabled; nothing to do")
        return 0

    output_csv = _resolve_output_path(args)
    config = MetricConfig(
        dataset_dir=args.dataset_dir,
        work_dir=args.work_dir,
        graph_dir=args.graph_dir,
        log_dir=args.log_dir,
        jobs=max(1, int(args.jobs)),
        contact_cutoff=float(args.contact_cutoff),
        output_csv=output_csv,
    )

    logger.info(
        "Parameters: dataset_dir=%s work_dir=%s graph_dir=%s output=%s cutoff=%.2f jobs=%d features=%s",
        config.dataset_dir,
        config.work_dir,
        config.graph_dir,
        config.output_csv,
        config.contact_cutoff,
        config.jobs,
        ", ".join(feature.name for feature in selected_features),
    )

    runner = GlobalMetricsRunner(config=config, features=selected_features, logger=logger)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
