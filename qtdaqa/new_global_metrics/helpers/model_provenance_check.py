#!/usr/bin/env python3
"""Recompute model provenance / filename tag features for validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_COMPARISON_TOLERANCE = 0.05
EPSILON = 1e-8


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("model_provenance_check")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)

    logger.propagate = False
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute filename provenance features per PDB.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("model_provenance.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDBs (0 = all)")
    parser.add_argument(
        "--compare-against",
        type=Path,
        default=None,
        help="Existing CSV containing model_provenance_* columns",
    )
    parser.add_argument(
        "--comparison-tolerance",
        type=float,
        default=DEFAULT_COMPARISON_TOLERANCE,
        help=f"Allowed absolute difference when comparing (default {DEFAULT_COMPARISON_TOLERANCE:g})",
    )
    parser.add_argument("--verbose", action="store_true")
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
        text = text[:-4]
    return text


def _tokenize(stem: str) -> List[str]:
    tokens = [tok for tok in re.split(r"[_\-]+", stem.lower()) if tok]
    return tokens or [stem.lower()]


def _entropy(tokens: List[str]) -> float:
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    token_count = len(tokens)
    probabilities = [count / token_count for count in counts.values()]
    return max(0.0, -sum(p * math.log(p + EPSILON) for p in probabilities))


def compute_provenance(pdb_path: Path) -> Dict[str, float]:
    stem = pdb_path.stem
    tokens = _tokenize(stem)
    token_count = len(tokens)

    digit_tokens = sum(token.isdigit() for token in tokens)
    alpha_tokens = sum(token.isalpha() for token in tokens)
    digit_fraction = digit_tokens / token_count
    alpha_fraction = alpha_tokens / token_count

    numeric_match = re.search(r"(\d+)$", stem)
    numeric_suffix = float(numeric_match.group(1)) if numeric_match else 0.0

    contains_af2 = float(any("af2" in token for token in tokens))
    parent_match = float(tokens[0] == pdb_path.parent.name.lower()) if tokens else 0.0
    variant_flag = float(any(re.match(r"[pu]\d+$", token) for token in tokens))

    entropy = float(_entropy(tokens))
    hash_value = int(hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF

    return {
        "model_provenance_token_count": float(token_count),
        "model_provenance_digit_token_fraction": float(digit_fraction),
        "model_provenance_alpha_token_fraction": float(alpha_fraction),
        "model_provenance_numeric_suffix": numeric_suffix,
        "model_provenance_contains_af2": contains_af2,
        "model_provenance_parent_match_flag": float(parent_match),
        "model_provenance_variant_flag": variant_flag,
        "model_provenance_token_entropy": entropy,
        "model_provenance_hash": float(hash_value),
    }


def _write_results(path: Path, rows: Sequence[Tuple[str, Dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_provenance_token_count",
        "model_provenance_digit_token_fraction",
        "model_provenance_alpha_token_fraction",
        "model_provenance_numeric_suffix",
        "model_provenance_contains_af2",
        "model_provenance_parent_match_flag",
        "model_provenance_variant_flag",
        "model_provenance_token_entropy",
        "model_provenance_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MODEL", *columns])
        for model, stats in rows:
            writer.writerow([model, *[f"{stats[col]:.6f}" for col in columns]])


def _load_reference(csv_path: Path) -> Dict[str, Tuple[float, ...]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, Tuple[float, ...]] = {}
        columns = [
            "model_provenance_token_count",
            "model_provenance_digit_token_fraction",
            "model_provenance_alpha_token_fraction",
            "model_provenance_numeric_suffix",
            "model_provenance_contains_af2",
            "model_provenance_parent_match_flag",
            "model_provenance_variant_flag",
            "model_provenance_token_entropy",
            "model_provenance_hash",
        ]
        for row in reader:
            model = row.get("MODEL")
            if not model:
                continue
            try:
                mapping[model] = tuple(float(row[col]) for col in columns)
            except ValueError:
                continue
    return mapping


def _compare(logger: logging.Logger, computed: Sequence[Tuple[str, Dict[str, float]]], reference: Dict[str, Tuple[float, ...]], tolerance: float) -> None:
    columns = [
        "model_provenance_token_count",
        "model_provenance_digit_token_fraction",
        "model_provenance_alpha_token_fraction",
        "model_provenance_numeric_suffix",
        "model_provenance_contains_af2",
        "model_provenance_parent_match_flag",
        "model_provenance_variant_flag",
        "model_provenance_token_entropy",
        "model_provenance_hash",
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
            logger.warning("Mismatch %s ours=%s theirs=%s", model, [round(stats[col], 4) for col in columns], [round(value, 4) for value in ref])
        if len(mismatches) > 10:
            logger.warning("Additional mismatches truncated (%d total)", len(mismatches))
    else:
        logger.info("All compared models matched the reference values")


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI entry
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
        stats = compute_provenance(pdb_path)
        rows.append((model, stats))
        if index % 20 == 0 or index == len(pdb_files):
            logger.info("Processed %d/%d models", index, len(pdb_files))

    _write_results(args.output_csv, rows)
    logger.info("Wrote model provenance features for %d models to %s", len(rows), args.output_csv)

    if args.compare_against:
        reference = _load_reference(args.compare_against)
        if not reference:
            logger.warning("Reference CSV %s is empty or missing expected columns", args.compare_against)
        else:
            _compare(logger, rows, reference, args.comparison_tolerance)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
