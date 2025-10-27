#!/usr/bin/env python3
"""Recalculate buried SASA using Biopython Shrake–Rupley for cross-validation."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio.PDB import PDBParser, ShrakeRupley

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_PROBE_RADIUS = 1.4
DEFAULT_SPHERE_POINTS = 100
DEFAULT_COMPARISON_TOLERANCE = 25.0  # Å^2


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("buried_sasa_check")
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
        description="Compute buried SASA per model via Biopython Shrake–Rupley.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing PDB decoys")
    parser.add_argument("--output-csv", type=Path, default=Path("buried_sasa.csv"))
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=DEFAULT_PROBE_RADIUS,
        help="Probe radius passed to Shrake–Rupley (Å)",
    )
    parser.add_argument(
        "--sphere-points",
        type=int,
        default=DEFAULT_SPHERE_POINTS,
        help="Number of points in the Shrake–Rupley spherical grid",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on processed PDBs (0 = no limit)",
    )
    parser.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="Existing CSV (MODEL, interface_buried_sasa) to compare against",
    )
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=DEFAULT_COMPARISON_TOLERANCE,
        help=f"Allowed absolute difference when comparing (default: {DEFAULT_COMPARISON_TOLERANCE:g} Å^2)",
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

def _sasa_for_entity(entity, probe_radius: float, sphere_points: int) -> float:
    calculator = ShrakeRupley(probe_radius=probe_radius, n_points=sphere_points)
    calculator.compute(entity, level="A")
    total = 0.0
    for atom in entity.get_atoms():
        value = getattr(atom, "sasa", None)
        if value is None:
            value = atom.xtra.get("EXP_SASA", 0.0)
        total += float(value or 0.0)
    return total


def compute_buried_sasa(pdb_path: Path, probe_radius: float, sphere_points: int) -> float:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return 0.0

    complex_area = _sasa_for_entity(model, probe_radius, sphere_points)

    monomer_area = 0.0
    for chain in model:
        monomer_area += _sasa_for_entity(chain, probe_radius, sphere_points)

    buried = monomer_area - complex_area
    return buried if buried > 0.0 else 0.0


def _write_results(path: Path, rows: Sequence[Tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MODEL", "interface_buried_sasa"])
        for model, value in rows:
            writer.writerow([model, f"{value:.4f}"])


def _load_reference(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, float] = {}
        for row in reader:
            model = row.get("MODEL")
            if not model:
                continue
            try:
                mapping[model] = float(row.get("interface_buried_sasa", "nan"))
            except ValueError:
                continue
    return mapping


def _compare_results(
    logger: logging.Logger,
    computed: Sequence[Tuple[str, float]],
    reference: Dict[str, float],
    tolerance: float,
) -> None:
    missing: List[str] = []
    mismatches: List[Tuple[str, float, float]] = []

    for model, value in computed:
        ref_value = reference.get(model)
        if ref_value is None:
            missing.append(model)
            continue
        if abs(ref_value - value) > tolerance:
            mismatches.append((model, value, ref_value))

    logger.info("Comparison tolerance in effect: ±%.4f Å^2", tolerance)
    if missing:
        logger.warning("Reference comparison missing %d model(s)", len(missing))
    if mismatches:
        logger.warning("Found %d mismatch(es) when comparing against reference CSV", len(mismatches))
        for model, ours, theirs in mismatches[:10]:
            logger.warning("Mismatch: %s ours=%.3f theirs=%.3f", model, ours, theirs)
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
        "Comparison tolerance configured (via --comparison-tolerance, default %.4f): ±%.4f Å^2",
        DEFAULT_COMPARISON_TOLERANCE,
        args.comparison_tolerance,
    )

    pdb_files = _discover_pdbs(args.dataset_dir)
    if args.limit > 0:
        pdb_files = pdb_files[: args.limit]
    if not pdb_files:
        logger.warning("No PDB files found under %s", args.dataset_dir)
        return 0

    rows: List[Tuple[str, float]] = []
    for index, pdb_path in enumerate(pdb_files, start=1):
        model = _model_identifier(args.dataset_dir, pdb_path)
        try:
            buried = compute_buried_sasa(pdb_path, args.probe_radius, args.sphere_points)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdb_path, exc)
            continue
        rows.append((model, buried))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote buried SASA values for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s was empty or missing expected columns", args.compare_against)
        else:
            _compare_results(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
