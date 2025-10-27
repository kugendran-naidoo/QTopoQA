#!/usr/bin/env python3
"""Recalculate sequence length/ratio aggregates for validation."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_COMPARISON_TOLERANCE = 0.05
EPSILON = 1e-8

from Bio.PDB import PDBParser


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("sequence_ratio_check")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sequence length ratio features per PDB.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing PDBs")
    parser.add_argument("--output-csv", type=Path, default=Path("sequence_ratio.csv"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of PDBs processed (0 = no limit)",
    )
    parser.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="Reference CSV with sequence_length_ratio columns",
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


def _lengths_biopython(pdb_path: Path) -> Dict[str, int]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("seq", str(pdb_path))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return {}

    lengths: Dict[str, int] = {}
    for chain in model:
        count = 0
        for residue in chain.get_residues():
            hetero = residue.id[0]
            if hetero.strip() not in {"H", "W"}:
                count += 1
        if count == 0:
            count = sum(1 for _ in chain.get_residues())
        chain_id = getattr(chain, "id", "") or f"chain_{len(lengths)}"
        if count > 0:
            lengths[chain_id] = count
    return lengths


def _lengths_from_text(pdb_path: Path) -> Dict[str, int]:
    lengths: Dict[str, set] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip() or " "
            res_seq = line[22:26].strip()
            icode = line[26].strip()
            key = (res_seq, icode)
            lengths.setdefault(chain_id, set()).add(key)
    return {chain: len(residues) for chain, residues in lengths.items() if residues}


def _aggregate(lengths: Dict[str, int]) -> Dict[str, float]:
    if not lengths:
        return {key: 0.0 for key in [
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
        ]}

    L = np.array(sorted(lengths.values(), reverse=True), dtype=float)
    k = len(L)
    T = float(L.sum())
    mu = float(L.mean())
    med = float(np.median(L))
    L1 = float(L[0])
    L2 = float(L[1]) if k > 1 else L1
    Lk = float(L[-1])

    feats = {
        "sequence_largest_fraction": L1 / (T + EPSILON),
        "sequence_top2_fraction": (L1 + (L2 if k > 1 else 0.0)) / (T + EPSILON),
        "sequence_log_L1_L2": float(np.log((L1 + EPSILON) / (L2 + EPSILON))),
        "sequence_max_min_ratio_clipped": float(min(L1 / (Lk + EPSILON), 10.0)),
        "sequence_cv": float(np.std(L) / (mu + EPSILON)),
        "sequence_median_length": med,
        "sequence_mean_length": mu,
        "sequence_p75_length": float(np.quantile(L, 0.75)),
        "sequence_n_long_ge_1p5med": float((L >= 1.5 * med).sum()),
        "sequence_n_short_le_0p5med": float((L <= 0.5 * med).sum()),
        "sequence_chain_count": float(k),
        "sequence_total_length": T,
    }
    return feats


def compute_sequence_ratio(pdb_path: Path) -> Dict[str, float]:
    lengths = _lengths_biopython(pdb_path)
    if not lengths:
        lengths = _lengths_from_text(pdb_path)
    return _aggregate(lengths)


def _write_results(path: Path, rows: Sequence[Tuple[str, Dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MODEL", *columns])
        for model, stats in rows:
            writer.writerow([model, *[f"{stats[key]:.6f}" for key in columns]])


def _load_reference(csv_path: Path) -> Dict[str, Tuple[float, ...]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, Tuple[float, ...]] = {}
        for row in reader:
            model = row.get("MODEL")
            if not model:
                continue
            try:
                mapping[model] = tuple(
                    float(row[name])
                    for name in [
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
                    ]
                )
            except ValueError:
                continue
    return mapping


def _compare_results(
    logger: logging.Logger,
    computed: Sequence[Tuple[str, Dict[str, float]]],
    reference: Dict[str, Tuple[float, ...]],
    tolerance: float,
) -> None:
    columns = [
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
    ]
    missing: List[str] = []
    mismatches: List[Tuple[str, Dict[str, float], Tuple[float, ...]]] = []

    for model, stats in computed:
        ref = reference.get(model)
        if ref is None:
            missing.append(model)
            continue
        values = tuple(stats[col] for col in columns)
        if any(abs(a - b) > tolerance for a, b in zip(values, ref)):
            mismatches.append((model, stats, ref))

    logger.info("Comparison tolerance in effect: ±%.4f", tolerance)
    if missing:
        logger.warning("Reference comparison missing %d model(s)", len(missing))
    if mismatches:
        logger.warning("Found %d mismatch(es) when comparing against reference CSV", len(mismatches))
        for model, stats, ref in mismatches[:10]:
            logger.warning("Mismatch %s ours=%s theirs=%s", model, [round(stats[col], 4) for col in columns], [round(val, 4) for val in ref])
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

    rows: List[Tuple[str, Dict[str, float]]] = []
    for index, pdb_path in enumerate(pdb_files, start=1):
        model = _model_identifier(args.dataset_dir, pdb_path)
        stats = compute_sequence_ratio(pdb_path)
        rows.append((model, stats))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote sequence length ratio aggregates for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s is empty or missing expected columns", args.compare_against)
        else:
            _compare_results(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
