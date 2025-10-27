#!/usr/bin/env python3
"""Recalculate interface contact counts using MDAnalysis for independent validation."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_CONTACT_CUTOFF = 5.0
DEFAULT_COMPARISON_TOLERANCE = 1.0

try:  # guard optional dependency
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
except Exception as exc:  # pragma: no cover - import guard
    MDANALYSIS_AVAILABLE = False
    MDANALYSIS_IMPORT_ERROR = exc
else:  # pragma: no cover - requires MDAnalysis runtime
    MDANALYSIS_AVAILABLE = True
    MDANALYSIS_IMPORT_ERROR = None


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("mdanalysis_contact_check")
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
        description="Recompute inter-chain heavy-atom contacts using MDAnalysis.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing PDB decoys")
    parser.add_argument("--output-csv", type=Path, default=Path("mdanalysis_contact_counts.csv"))
    parser.add_argument("--contact-cutoff", type=float, default=DEFAULT_CONTACT_CUTOFF, help="Heavy-atom cutoff distance (Å)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of PDBs to process (0 disables the limit)",
    )
    parser.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="Existing CSV (MODEL, interface_contact_count) to compare against",
    )
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=DEFAULT_COMPARISON_TOLERANCE,
        help=f"Allowed absolute difference in contact counts when comparing (default: {DEFAULT_COMPARISON_TOLERANCE:g})",
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
    stem = relative.as_posix()
    if stem.lower().endswith(".pdb"):
        stem = stem[: -len(".pdb")]
    return stem


def _load_reference(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, float] = {}
        for row in reader:
            model = row.get("MODEL")
            if not model:
                continue
            try:
                value = float(row.get("interface_contact_count", "nan"))
            except ValueError:
                continue
            mapping[model] = value
    return mapping


def _iter_chain_pairs(universe: "mda.Universe") -> Iterable[Tuple["mda.AtomGroup", "mda.AtomGroup"]]:
    groups: List["mda.AtomGroup"] = []
    for chain_id, atom_group in universe.atoms.groupby("chainIDs").items():
        heavy_atoms = atom_group.select_atoms("not name H*")
        if len(heavy_atoms):
            groups.append(heavy_atoms)
    return combinations(groups, 2)


def compute_contact_count(pdb_path: Path, cutoff: float) -> int:
    universe = mda.Universe(str(pdb_path))
    total = 0
    for atoms_a, atoms_b in _iter_chain_pairs(universe):
        pairs = capped_distance(atoms_a.positions, atoms_b.positions, cutoff, return_distances=False)
        if isinstance(pairs, tuple):
            total += len(pairs[0])
        else:
            total += len(pairs)
    return total


@dataclass
class ResultRow:
    model: str
    contact_count: float


def _write_results(path: Path, rows: Sequence[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MODEL", "interface_contact_count"])
        for row in rows:
            writer.writerow([row.model, f"{row.contact_count:.0f}"])


def _compare_results(
    logger: logging.Logger,
    computed: Sequence[ResultRow],
    reference: Dict[str, float],
    tolerance: float,
) -> None:
    missing: List[str] = []
    mismatches: List[Tuple[str, float, float]] = []

    for row in computed:
        ref_value = reference.get(row.model)
        if ref_value is None:
            missing.append(row.model)
            continue
        if abs(ref_value - row.contact_count) > tolerance:
            mismatches.append((row.model, row.contact_count, ref_value))

    logger.info("Comparison tolerance in effect: ±%.4f contact(s)", tolerance)
    if missing:
        logger.warning("Reference comparison missing %d model(s)", len(missing))
    if mismatches:
        logger.warning("Found %d mismatch(es) when comparing against reference CSV", len(mismatches))
        for model, ours, theirs in mismatches[:10]:
            logger.warning("Mismatch: %s ours=%.0f theirs=%.0f", model, ours, theirs)
        if len(mismatches) > 10:
            logger.warning("Additional mismatches truncated (%d total)", len(mismatches))
    else:
        logger.info("All compared models matched the reference values")


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)

    if not MDANALYSIS_AVAILABLE:
        print(
            f"MDAnalysis is required for this helper script but could not be imported: {MDANALYSIS_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 2

    logger = _configure_logger(args.verbose)

    if not args.dataset_dir.exists():
        logger.error("Dataset directory does not exist: %s", args.dataset_dir)
        return 1

    logger.info(
        "Comparison tolerance configured (via --comparison-tolerance, default %.4f): ±%.4f contact(s)",
        DEFAULT_COMPARISON_TOLERANCE,
        args.comparison_tolerance,
    )

    pdb_files = _discover_pdbs(args.dataset_dir)
    if args.limit > 0:
        pdb_files = pdb_files[: args.limit]
    if not pdb_files:
        logger.warning("No PDB files found under %s", args.dataset_dir)
        return 0

    rows: List[ResultRow] = []
    for index, pdb_path in enumerate(pdb_files, start=1):
        model = _model_identifier(args.dataset_dir, pdb_path)
        try:
            count = compute_contact_count(pdb_path, args.contact_cutoff)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdb_path, exc)
            continue
        rows.append(ResultRow(model=model, contact_count=float(count)))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote MDAnalysis counts for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s was empty or missing expected columns", args.compare_against)
        else:
            _compare_results(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
