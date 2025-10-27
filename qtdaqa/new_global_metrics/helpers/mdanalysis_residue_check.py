#!/usr/bin/env python3
"""Recalculate unique interface residue counts using MDAnalysis for validation."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_CONTACT_CUTOFF = 5.0
DEFAULT_COMPARISON_TOLERANCE = 1.0

try:  # pragma: no cover - import guard
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import capped_distance
except Exception as exc:  # pragma: no cover - import guard
    MDANALYSIS_AVAILABLE = False
    MDANALYSIS_IMPORT_ERROR = exc
else:  # pragma: no cover - requires MDAnalysis runtime
    MDANALYSIS_AVAILABLE = True
    MDANALYSIS_IMPORT_ERROR = None


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("mdanalysis_residue_check")
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
        description="Recompute inter-chain interface residue counts using MDAnalysis.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing PDB decoys")
    parser.add_argument("--output-csv", type=Path, default=Path("residue_counts.csv"))
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=DEFAULT_CONTACT_CUTOFF,
        help="Heavy-atom cutoff distance for defining contacts (Å)",
    )
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
        help="Existing CSV (MODEL, interface_residue_count) to compare against",
    )
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=DEFAULT_COMPARISON_TOLERANCE,
        help=f"Allowed absolute difference in residue counts when comparing (default: {DEFAULT_COMPARISON_TOLERANCE:g})",
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
    name = relative.as_posix()
    if name.lower().endswith(".pdb"):
        name = name[: -len(".pdb")]
    return name


def _iter_chain_pairs(universe: "mda.Universe") -> Iterable[Tuple["mda.AtomGroup", "mda.AtomGroup"]]:
    groups: List["mda.AtomGroup"] = []
    for _, atom_group in universe.atoms.groupby("chainIDs").items():
        heavy = atom_group.select_atoms("not name H*")
        if len(heavy):
            groups.append(heavy)
    return combinations(groups, 2)


def _residue_descriptor(atom) -> Tuple[str, int, str]:
    chain = atom.chainID.strip() or atom.segid.strip()
    return chain, int(atom.resid), atom.resname.strip()


def compute_residue_count(pdb_path: Path, cutoff: float) -> int:
    universe = mda.Universe(str(pdb_path))
    residues: set[Tuple[str, int, str]] = set()
    for atoms_a, atoms_b in _iter_chain_pairs(universe):
        if not len(atoms_a) or not len(atoms_b):
            continue
        descriptors_a = [_residue_descriptor(atom) for atom in atoms_a]
        descriptors_b = [_residue_descriptor(atom) for atom in atoms_b]
        pairs = capped_distance(atoms_a.positions, atoms_b.positions, cutoff, return_distances=False)
        if isinstance(pairs, tuple):
            idx_a, idx_b = pairs
        else:
            idx_a, idx_b = pairs[:, 0], pairs[:, 1]
        for ia, ib in zip(idx_a, idx_b):
            residues.add(descriptors_a[int(ia)])
            residues.add(descriptors_b[int(ib)])
    return len(residues)


def _load_reference(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, float] = {}
        for row in reader:
            model = row.get("MODEL")
            if not model:
                continue
            try:
                mapping[model] = float(row.get("interface_residue_count", "nan"))
            except ValueError:
                continue
    return mapping


def _write_results(path: Path, rows: Sequence[Tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MODEL", "interface_residue_count"])
        for model, value in rows:
            writer.writerow([model, f"{value:.0f}"])


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

    logger.info("Comparison tolerance in effect: ±%.4f residue(s)", tolerance)
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
        "Comparison tolerance configured (via --comparison-tolerance, default %.4f): ±%.4f residue(s)",
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
            count = compute_residue_count(pdb_path, args.contact_cutoff)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdb_path, exc)
            continue
        rows.append((model, float(count)))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote MDAnalysis residue counts for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s was empty or missing expected columns", args.compare_against)
        else:
            _compare_results(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
