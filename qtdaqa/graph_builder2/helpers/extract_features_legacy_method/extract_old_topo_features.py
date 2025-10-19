#!/usr/bin/env python3
"""Extract topology features using the legacy topo_feature implementation."""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Error: pandas is required to run this script.") from exc

try:
    from Bio.PDB import PDBParser
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Error: biopython is required to run this script.") from exc


def _locate_repo_root(start: Path) -> Path:
    current = start
    for parent in [current] + list(current.parents):
        candidate = parent / "topoqa" / "src"
        if candidate.exists():
            return parent
    raise RuntimeError("Unable to locate repo root containing 'topoqa/src'.")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
TOPOQA_SRC = REPO_ROOT / "topoqa" / "src"
if str(TOPOQA_SRC) not in sys.path:
    sys.path.insert(0, str(TOPOQA_SRC))

from topo_feature import topo_fea  # type: ignore  # noqa: E402
from get_interface import cal_interface  # type: ignore  # noqa: E402

ElementFilter = Union[List[str], str]
DEFAULT_ELEMENT_TOKENS: List[str] = ["C", "N", "O", "CN", "CO", "NO", "CNO"]
DEFAULT_ELEMENT_FILTERS: List[List[str]] = [
    ["C"],
    ["N"],
    ["O"],
    ["C", "N"],
    ["C", "O"],
    ["N", "O"],
    ["C", "N", "O"],
]
INTERFACE_COORD_DECIMALS = 3


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text before exiting on errors."""

    def error(self, message: str) -> None:  # pragma: no cover
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def _normalise_element_filters(values: Optional[Iterable[str]]) -> List[ElementFilter]:
    if not values:
        return [group.copy() for group in DEFAULT_ELEMENT_FILTERS]
    cleaned: List[ElementFilter] = []
    for token in values:
        trimmed = token.strip()
        if not trimmed:
            continue
        if trimmed.lower() == "all":
            return ["all"]
        letters = [char.upper() for char in trimmed if char.isalpha()]
        if not letters:
            continue
        cleaned.append(list(dict.fromkeys(letters)))
    return cleaned or ["all"]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing input structures (.pdb). (required)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write topology CSV files ('.topology.csv'). (required)",
    )
    parser.add_argument(
        "--neighbor-distance",
        type=float,
        default=8.0,
        help="Neighborhood radius in Å for point cloud extraction.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Persistence cutoff for H0 bars.",
    )
    parser.add_argument(
        "--interface-cutoff",
        type=float,
        default=10.0,
        help="Interface detection cutoff in Å (used when selecting residues).",
    )
    parser.add_argument(
        "--min-persistence",
        type=float,
        default=0.01,
        help="Minimum persistence threshold to keep bars.",
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=list(DEFAULT_ELEMENT_TOKENS),
        help="Element filters (use tokens like C, N, O, CN, ... or 'all').",
    )
    parser.add_argument(
        "--residues-per-file",
        type=int,
        default=0,
        help="If >0, limit to this many residues per structure (0 = all).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes (1 = sequential).",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional limit on number of models to process (0 = all).",
    )
    return parser.parse_args(list(argv))


@dataclass
class ExtractionTask:
    model: str
    pdb_path: Path
    interface_output: Path
    output_path: Path


def _collect_tasks(dataset_dir: Path, output_dir: Path, max_models: int) -> List[ExtractionTask]:
    pdb_files = sorted(dataset_dir.rglob("*.pdb"))
    tasks: List[ExtractionTask] = []
    for pdb_path in pdb_files:
        model = pdb_path.stem
        interface_placeholder = output_dir / "interface_tmp" / f"{model}.interface.txt"
        topology_output = output_dir / f"{model}.topology.csv"
        topology_output.parent.mkdir(parents=True, exist_ok=True)
        interface_placeholder.parent.mkdir(parents=True, exist_ok=True)
        tasks.append(
            ExtractionTask(
                model=model,
                pdb_path=pdb_path,
                interface_output=interface_placeholder,
                output_path=topology_output,
            )
        )
        if max_models and len(tasks) >= max_models:
            break
    return tasks


def _generate_interface(task: ExtractionTask, cutoff: float) -> List[str]:
    extractor = cal_interface(str(task.pdb_path), cut=cutoff)
    extractor.find_and_write(str(task.interface_output))
    _round_interface_file(task.interface_output, INTERFACE_COORD_DECIMALS)
    residues: List[str] = []
    with task.interface_output.open() as handle:
        for line in handle:
            token = line.strip().split()[0]
            residues.append(token)
    return residues


def _process_task(
    task: ExtractionTask,
    element_filters: List[ElementFilter],
    neighbor_distance: float,
    cutoff: float,
    min_persistence: float,
    residues_per_file: int,
    interface_cutoff: float,
) -> Tuple[str, Optional[str]]:
    temp_interface = None
    try:
        residues = _generate_interface(task, interface_cutoff)
        if residues_per_file > 0:
            residues = residues[:residues_per_file]
        if not residues:
            return task.model, "no interface residues found"
        extractor = topo_fea(
            str(task.pdb_path),
            neighbor_dis=neighbor_distance,
            e_set=element_filters,
            res_list=residues,
            Cut=cutoff,
        )
        df = extractor.cal_fea()
        pd.set_option("future.no_silent_downcasting", True)
        df.replace("NA", pd.NA, inplace=True)
        df.to_csv(task.output_path, index=False)
        task.interface_output.unlink(missing_ok=True)
        return task.model, None
    except Exception as exc:  # pragma: no cover
        return task.model, str(exc)


def run_extraction(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("Configuration:")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  neighbor_distance: {args.neighbor_distance}")
    print(f"  cutoff: {args.cutoff}")
    print(f"  min_persistence: {args.min_persistence}")
    print(f"  elements: {args.elements}")
    print(f"  residues_per_file: {args.residues_per_file}")
    print(f"  interface_cutoff: {args.interface_cutoff}")
    print(f"  jobs: {args.jobs}  # number of worker processes (1 = sequential)")
    print(f"  max_models: {args.max_models}  # optional limit on models to process (0 = all)")

    if not dataset_dir.is_dir():
        print(f"[error] dataset directory not found: {dataset_dir}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    element_filters = _normalise_element_filters(args.elements)
    tasks = _collect_tasks(dataset_dir, output_dir, args.max_models)
    if not tasks:
        print("[warn] No '.pdb' files found; nothing to do.")
        return 0

    success = 0
    failures: List[Tuple[str, str]] = []

    if args.jobs and args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_map = {
                executor.submit(
                    _process_task,
                    task,
                    element_filters,
                    args.neighbor_distance,
                    args.cutoff,
                    args.min_persistence,
                    args.residues_per_file,
                    args.interface_cutoff,
                ): task.model
                for task in tasks
            }
            for future in as_completed(future_map):
                model, error = future.result()
                if error:
                    failures.append((model, error))
                    print(f"[warn] {model}: {error}")
                else:
                    success += 1
    else:
        for task in tasks:
            model, error = _process_task(
                task,
                element_filters,
                args.neighbor_distance,
                args.cutoff,
                args.min_persistence,
                args.residues_per_file,
                args.interface_cutoff,
            )
            if error:
                failures.append((model, error))
                print(f"[warn] {model}: {error}")
            else:
                success += 1

    print("Summary:")
    print(f"  processed: {len(tasks)}")
    print(f"  succeeded: {success}")
    print(f"  failed:    {len(failures)}")
    if failures:
        print("Failures:")
        for model, error in failures:
            print(f"  {model}: {error}")
        return 1
    return 0


def describe_source() -> None:
    print("This script reuses code from topoqa/src/topo_feature.py (topo_fea class).")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    return run_extraction(args)


if __name__ == "__main__":
    describe_source()
    raise SystemExit(main(sys.argv[1:]))


def _round_interface_file(path: Path, decimals: int) -> None:
    """Normalise interface coordinate precision to match legacy inference outputs."""
    if decimals < 0:
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    format_str = f"{{:.{decimals}f}}"
    updated: List[str] = []
    changed = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            updated.append("")
            continue
        parts = stripped.split()
        if len(parts) < 4:
            updated.append(stripped)
            continue
        descriptor, coords = parts[0], parts[1:]
        try:
            rounded = [format_str.format(float(value)) for value in coords]
        except ValueError:
            updated.append(stripped)
            continue
        combined = " ".join([descriptor, *rounded])
        if combined != stripped:
            changed = True
        updated.append(combined)

    if changed:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(updated))
            handle.write("\n")
