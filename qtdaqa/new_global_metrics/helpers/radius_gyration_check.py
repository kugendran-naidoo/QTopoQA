#!/usr/bin/env python3
"""Recalculate chain radius-of-gyration aggregates for validation."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_COMPARISON_TOLERANCE = 0.05
EPSILON = 1e-6

try:
    import MDAnalysis as mda

    HAS_MDANALYSIS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_MDANALYSIS = False

from Bio.PDB import PDBParser


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("radius_gyration_check")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute chain radius-of-gyration aggregates per PDB.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing PDB decoys")
    parser.add_argument("--output-csv", type=Path, default=Path("radius_of_gyration.csv"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many PDB files (0 = no limit)",
    )
    parser.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="Reference CSV with chain_radius_of_gyration_* columns",
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


def _model_identifier(dataset_dir: Path, pdb_path: Path) -> str:
    try:
        relative = pdb_path.relative_to(dataset_dir)
    except ValueError:
        relative = pdb_path.name
    text = relative.as_posix()
    if text.lower().endswith(".pdb"):
        text = text[: -len(".pdb")]
    return text


def _radius_of_gyration(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 0.0
    centroid = np.mean(coords, axis=0)
    diffs = coords - centroid
    squared = np.sum(diffs * diffs, axis=1)
    return float(np.sqrt(max(EPSILON, np.mean(squared))))


def _compute_mdanalysis(pdb_path: Path) -> List[float]:
    universe = mda.Universe(str(pdb_path))
    values: List[float] = []
    for chain_id, atom_group in universe.atoms.groupby("chainIDs").items():
        heavy = atom_group.select_atoms("not name H*")
        coords = heavy.positions if len(heavy) else atom_group.positions
        if coords.size == 0:
            continue
        values.append(_radius_of_gyration(np.asarray(coords, dtype=float)))
    return values


def _compute_biopython(pdb_path: Path) -> List[float]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rg", str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return []

    values: List[float] = []
    for chain in model:
        coords = []
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
            continue
        values.append(_radius_of_gyration(np.vstack(coords)))
    return values


def compute_chain_rg(pdb_path: Path) -> Dict[str, float]:
    values = _compute_mdanalysis(pdb_path) if HAS_MDANALYSIS else _compute_biopython(pdb_path)
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    mean_value = float(np.mean(values))
    median_value = float(np.median(values))
    std_value = float(np.std(values)) if len(values) > 1 else 0.0
    return {"mean": mean_value, "median": median_value, "std": std_value}


def _write_results(path: Path, rows: Sequence[Tuple[str, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "MODEL",
                "chain_radius_of_gyration_mean",
                "chain_radius_of_gyration_median",
                "chain_radius_of_gyration_std",
            ]
        )
        for model, mean_value, median_value, std_value in rows:
            writer.writerow([model, f"{mean_value:.6f}", f"{median_value:.6f}", f"{std_value:.6f}"])


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
                    float(row.get("chain_radius_of_gyration_mean", "nan")),
                    float(row.get("chain_radius_of_gyration_median", "nan")),
                    float(row.get("chain_radius_of_gyration_std", "nan")),
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

    for model, mean_value, median_value, std_value in computed:
        ref = reference.get(model)
        if ref is None:
            missing.append(model)
            continue
        if (
            abs(ref[0] - mean_value) > tolerance
            or abs(ref[1] - median_value) > tolerance
            or abs(ref[2] - std_value) > tolerance
        ):
            mismatches.append((model, (mean_value, median_value, std_value), ref))

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
        model = _model_identifier(args.dataset_dir, pdb_path)
        stats = compute_chain_rg(pdb_path)
        rows.append((model, stats["mean"], stats["median"], stats["std"]))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote chain radius-of-gyration aggregates for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s is empty or missing expected columns", args.compare_against)
        else:
            _compare_results(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
