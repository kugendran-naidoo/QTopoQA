#!/usr/bin/env python3
"""Extract interface features using the legacy cal_interface implementation."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Ensure legacy topoqa code is importable
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

from get_interface import cal_interface  # type: ignore  # noqa: E402


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that shows full help text before exiting on error."""

    def error(self, message: str) -> None:  # pragma: no cover - CLI convenience
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


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
        help="Directory to write interface text files ('.interface.txt'). (required)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="Distance cutoff (Å) for interface detection.",
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
        help="Optional limit on the number of models to process (0 = all).",
    )
    return parser.parse_args(list(argv))


@dataclass
class ExtractionTask:
    model: str
    pdb_path: Path
    output_path: Path


def _collect_tasks(dataset_dir: Path, output_dir: Path, max_models: int) -> list[ExtractionTask]:
    pdb_files = sorted(dataset_dir.rglob("*.pdb"))
    tasks: list[ExtractionTask] = []
    for pdb_path in pdb_files:
        model = pdb_path.stem
        target_output = output_dir / f"{model}.interface.txt"
        target_output.parent.mkdir(parents=True, exist_ok=True)
        tasks.append(ExtractionTask(model=model, pdb_path=pdb_path, output_path=target_output))
        if max_models and len(tasks) >= max_models:
            break
    return tasks


def _process_task(task: ExtractionTask, cutoff: float) -> tuple[str, Optional[str]]:
    try:
        interface_calc = cal_interface(str(task.pdb_path), cut=cutoff)
        interface_calc.find_and_write(str(task.output_path))
        return task.model, None
    except Exception as exc:  # pragma: no cover - operational logging
        return task.model, str(exc)


def run_extraction(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("Configuration:")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  cutoff: {args.cutoff}")
    print(f"  jobs: {args.jobs}  # number of worker processes (1 = sequential)")
    print(f"  max_models: {args.max_models}  # optional limit on models to process (0 = all)")

    if not dataset_dir.is_dir():
        print(f"[error] dataset directory not found: {dataset_dir}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(dataset_dir, output_dir, args.max_models)
    if not tasks:
        print("[warn] No '.pdb' files found; nothing to do.")
        return 0

    if args.jobs and args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        success = 0
        failures: list[tuple[str, str]] = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_map = {
                executor.submit(_process_task, task, args.cutoff): task.model for task in tasks
            }
            for future in as_completed(future_map):
                model, error = future.result()
                if error:
                    failures.append((model, error))
                    print(f"[warn] {model}: {error}")
                else:
                    success += 1
    else:
        success = 0
        failures = []
        for task in tasks:
            model, error = _process_task(task, args.cutoff)
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
    print("This script reuses code from topoqa/src/get_interface.py (cal_interface class).")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    return run_extraction(args)


if __name__ == "__main__":
    describe_source()
    raise SystemExit(main(sys.argv[1:]))
