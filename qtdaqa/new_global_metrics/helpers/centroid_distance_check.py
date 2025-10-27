#!/usr/bin/env python3
"""Recalculate interface centroid distance aggregates for cross-validation."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_CONTACT_CUTOFF = 5.0
DEFAULT_SOFTMIN_ALPHA = 2.0
DEFAULT_COMPARISON_TOLERANCE = 0.05
EPSILON = 1e-6

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MDANALYSIS = False


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("centroid_distance_check")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute interface centroid distance aggregates for each decoy.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing PDB decoys")
    parser.add_argument("--output-csv", type=Path, default=Path("centroid_distance.csv"))
    parser.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF, help="Heavy-atom contact cutoff (Å)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N PDBs (0 = no limit)",
    )
    parser.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="Reference CSV with columns interface_centroid_distance_* for comparison",
    )
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=DEFAULT_COMPARISON_TOLERANCE,
        help=f"Allowed absolute difference when comparing (default {DEFAULT_COMPARISON_TOLERANCE:g})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def _discover_pdbs(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.pdb") if path.is_file())


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resid: int
    icode: str
    resname: str

    def to_biopython(self) -> Tuple[str, int, str]:
        return self.chain, self.resid, self.icode or " "


def _model_identifier(dataset_dir: Path, pdb_path: Path) -> str:
    try:
        relative = pdb_path.relative_to(dataset_dir)
    except ValueError:
        relative = pdb_path.name
    text = relative.as_posix()
    if text.lower().endswith(".pdb"):
        text = text[: -len(".pdb")]
    return text


def _collect_pairs_mdanalysis(pdb_path: Path, cutoff: float) -> Dict[Tuple[str, str], Dict[str, object]]:
    universe = mda.Universe(str(pdb_path))
    atoms_by_chain = {cid: group for cid, group in universe.atoms.groupby("chainIDs").items() if len(group)}
    chain_ids = sorted(atoms_by_chain)

    pair_map: Dict[Tuple[str, str], Dict[str, object]] = {}

    for idx, chain_a in enumerate(chain_ids):
        atoms_a = atoms_by_chain[chain_a].select_atoms("not name H*")
        if not len(atoms_a):
            continue
        for chain_b in chain_ids[idx + 1 :]:
            atoms_b = atoms_by_chain[chain_b].select_atoms("not name H*")
            if not len(atoms_b):
                continue
            pairs = capped_distance(atoms_a.positions, atoms_b.positions, cutoff, return_distances=False)
            if isinstance(pairs, tuple):
                idx_a, idx_b = pairs
            else:  # pragma: no cover - legacy API
                idx_a, idx_b = pairs[:, 0], pairs[:, 1]
            if len(idx_a) == 0:
                continue
            record = pair_map.setdefault(
                (chain_a, chain_b),
                {"contacts": 0, "residues": {chain_a: set(), chain_b: set()}},
            )
            residues: Dict[str, Set[ResidueKey]] = record["residues"]  # type: ignore[assignment]
            for ia, ib in zip(idx_a, idx_b):
                atom_a = atoms_a[int(ia)]
                atom_b = atoms_b[int(ib)]
                icode_a = getattr(atom_a, "insertion_code", getattr(atom_a, "icode", ""))
                icode_b = getattr(atom_b, "insertion_code", getattr(atom_b, "icode", ""))
                res_a = ResidueKey(
                    chain=chain_a,
                    resid=int(atom_a.resid),
                    icode=(icode_a or "").strip(),
                    resname=atom_a.resname.strip(),
                )
                res_b = ResidueKey(
                    chain=chain_b,
                    resid=int(atom_b.resid),
                    icode=(icode_b or "").strip(),
                    resname=atom_b.resname.strip(),
                )
                residues.setdefault(chain_a, set()).add(res_a)
                residues.setdefault(chain_b, set()).add(res_b)
            record["contacts"] = int(record["contacts"]) + len(idx_a)

    return pair_map


def _collect_pairs_biopython(pdb_path: Path, cutoff: float) -> Dict[Tuple[str, str], Dict[str, object]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return {}

    atoms = [atom for atom in model.get_atoms() if (atom.element or "").strip().upper() != "H"]
    search = NeighborSearch(atoms)

    pair_map: Dict[Tuple[str, str], Dict[str, object]] = {}

    for atom in atoms:
        residue = atom.get_parent()
        chain = residue.get_parent()
        chain_id = getattr(chain, "id", "")
        neighbors = search.search(atom.coord, cutoff)
        for other in neighbors:
            other_res = other.get_parent()
            other_chain = other_res.get_parent()
            other_chain_id = getattr(other_chain, "id", "")
            if chain_id >= other_chain_id:
                continue
            record = pair_map.setdefault(
                (chain_id, other_chain_id),
                {"contacts": 0, "residues": {chain_id: set(), other_chain_id: set()}},
            )
            residues: Dict[str, Set[ResidueKey]] = record["residues"]  # type: ignore[assignment]
            residues.setdefault(chain_id, set()).add(
                ResidueKey(
                    chain=chain_id,
                    resid=int(residue.id[1]),
                    icode=(residue.id[2] or "").strip(),
                    resname=residue.get_resname().strip(),
                )
            )
            residues.setdefault(other_chain_id, set()).add(
                ResidueKey(
                    chain=other_chain_id,
                    resid=int(other_res.id[1]),
                    icode=(other_res.id[2] or "").strip(),
                    resname=other_res.get_resname().strip(),
                )
            )
            record["contacts"] = int(record["contacts"]) + 1

    return pair_map


def _residue_atom_coords(structure, residue_key: ResidueKey) -> Optional[np.ndarray]:
    chain_id, resid, icode = residue_key.to_biopython()
    try:
        chain = structure[chain_id]
    except KeyError:
        return None
    try:
        residue = chain[(" ", resid, icode or " ")]
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


def _chain_coords(structure, residue_keys: Set[ResidueKey]) -> Optional[np.ndarray]:
    blocks: List[np.ndarray] = []
    for key in residue_keys:
        coords = _residue_atom_coords(structure, key)
        if coords is not None:
            blocks.append(coords)
    if not blocks:
        return None
    return np.vstack(blocks)


def _radius_of_gyration(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 0.0
    centroid = np.mean(coords, axis=0)
    diffs = coords - centroid
    return float(np.sqrt(max(EPSILON, np.mean(np.sum(diffs * diffs, axis=1)))))


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


def _compute_aggregates(pair_map: Dict[Tuple[str, str], Dict[str, object]], structure) -> Dict[str, float]:
    if not pair_map:
        return {"median": 0.0, "softmin": 0.0, "weighted_mean": 0.0}

    values: List[float] = []
    weights: List[float] = []

    for (chain_a, chain_b), record in pair_map.items():
        residues: Dict[str, Set[ResidueKey]] = record["residues"]  # type: ignore[assignment]
        coords_a = _chain_coords(structure, residues.get(chain_a, set()))
        coords_b = _chain_coords(structure, residues.get(chain_b, set()))
        if coords_a is None or coords_b is None:
            continue
        centroid_a = np.mean(coords_a, axis=0)
        centroid_b = np.mean(coords_b, axis=0)
        distance = float(np.linalg.norm(centroid_a - centroid_b))
        rg_a = _radius_of_gyration(coords_a)
        rg_b = _radius_of_gyration(coords_b)
        scale = max(EPSILON, (rg_a + rg_b) / 2.0)
        values.append(distance / scale)
        weights.append(float(max(1, int(record["contacts"]))))

    if not values:
        return {"median": 0.0, "softmin": 0.0, "weighted_mean": 0.0}

    median_value = float(np.median(values))
    softmin_value = float(_softmin(values, DEFAULT_SOFTMIN_ALPHA))
    weight_sum = sum(weights)
    weighted_mean = float(sum(v * w for v, w in zip(values, weights)) / weight_sum)

    return {
        "median": median_value,
        "softmin": softmin_value,
        "weighted_mean": weighted_mean,
    }


def compute_centroid_distances(pdb_path: Path, cutoff: float) -> Dict[str, float]:
    if HAS_MDANALYSIS:
        pair_map = _collect_pairs_mdanalysis(pdb_path, cutoff)
    else:  # pragma: no cover - MDAnalysis unavailable
        pair_map = _collect_pairs_biopython(pdb_path, cutoff)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return {"median": 0.0, "softmin": 0.0, "weighted_mean": 0.0}

    return _compute_aggregates(pair_map, model)


def _write_results(path: Path, rows: Sequence[Tuple[str, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "MODEL",
                "interface_centroid_distance_median",
                "interface_centroid_distance_softmin",
                "interface_centroid_distance_weighted_mean",
            ]
        )
        for model, median, softmin, weighted in rows:
            writer.writerow([model, f"{median:.6f}", f"{softmin:.6f}", f"{weighted:.6f}"])


def _load_reference(csv_path: Path) -> Dict[str, Tuple[float, float, float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, Tuple[float, float, float]] = {}
        for row in reader:
            model = row.get("MODEL")
            if not model:
                continue
            try:
                mapping[model] = (
                    float(row.get("interface_centroid_distance_median", "nan")),
                    float(row.get("interface_centroid_distance_softmin", "nan")),
                    float(row.get("interface_centroid_distance_weighted_mean", "nan")),
                )
            except ValueError:
                continue
    return mapping


def _compare_results(
    logger: logging.Logger,
    computed: Sequence[Tuple[str, float, float, float]],
    reference: Dict[str, Tuple[float, float, float]],
    tolerance: float,
) -> None:
    missing: List[str] = []
    mismatches: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]] = []

    for model, median, softmin, weighted in computed:
        ref = reference.get(model)
        if ref is None:
            missing.append(model)
            continue
        if (
            abs(ref[0] - median) > tolerance
            or abs(ref[1] - softmin) > tolerance
            or abs(ref[2] - weighted) > tolerance
        ):
            mismatches.append((model, (median, softmin, weighted), ref))

    logger.info("Comparison tolerance in effect: ±%.4f", tolerance)
    if missing:
        logger.warning("Reference comparison missing %d model(s)", len(missing))
    if mismatches:
        logger.warning("Found %d mismatch(es) when comparing against reference CSV", len(mismatches))
        for model, ours, ref in mismatches[:10]:
            logger.warning(
                "Mismatch %s ours=(%.4f, %.4f, %.4f) theirs=(%.4f, %.4f, %.4f)",
                model,
                *ours,
                *ref,
            )
        if len(mismatches) > 10:
            logger.warning("Additional mismatches truncated (%d total)", len(mismatches))
    else:
        logger.info("All compared models matched the reference values")


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)
    logger = _configure_logger(args.verbose)

    if not args.dataset_dir.exists():
        logger.error("Dataset directory does not exist: %s", args.dataset_dir)
        return 1

    logger.info(
        "Comparison tolerance configured (via --comparison-tolerance, default %.4f): ±%.4f",
        DEFAULT_COMPARISON_TOLERANCE,
        args.comparison_tolerance,
    )

    pdb_files = _discover_pdbs(args.dataset_dir)
    if args.limit > 0:
        pdb_files = pdb_files[: args.limit]
    if not pdb_files:
        logger.warning("No PDB files found under %s", args.dataset_dir)
        return 0

    rows: List[Tuple[str, float, float, float]] = []
    for index, pdb_path in enumerate(pdb_files, start=1):
        model_name = _model_identifier(args.dataset_dir, pdb_path)
        stats = compute_centroid_distances(pdb_path, args.contact_cutoff)
        rows.append((model_name, stats["median"], stats["softmin"], stats["weighted_mean"]))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote centroid distance aggregates for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s is empty or missing expected columns", args.compare_against)
        else:
            _compare_results(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
