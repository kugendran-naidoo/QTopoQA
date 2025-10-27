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
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from Bio.PDB import NeighborSearch, PDBParser
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


@dataclass
class ModelContext:
    pdb_path: Path
    structure: Structure
    contact_cutoff: float
    _contact_summary: Optional[ContactSummary] = field(default=None, init=False, repr=False)

    def get_contact_summary(self) -> ContactSummary:
        if self._contact_summary is None:
            self._contact_summary = _build_contact_summary(self.structure, self.contact_cutoff)
        return self._contact_summary


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


FEATURES: Tuple[MetricFeature, ...] = (
    InterfaceContactCountFeature(),
    InterfaceResidueCountFeature(),
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
