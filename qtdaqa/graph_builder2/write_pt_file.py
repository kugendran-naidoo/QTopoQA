#!/usr/bin/env python3
"""Build graph .pt files using legacy interface, topology, and node feature outputs."""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import torch  # noqa: F401  - required for create_graph
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Error: torch is required to write .pt files.") from exc


def _locate_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        candidate = parent / "topoqa" / "src"
        if candidate.exists():
            return parent
    raise RuntimeError("Unable to locate repo root containing 'topoqa/src'.")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
TOPOQA_DIR = REPO_ROOT / "topoqa"
if str(TOPOQA_DIR) not in sys.path:
    sys.path.insert(0, str(TOPOQA_DIR))

from src.graph import create_graph  # type: ignore  # noqa: E402

DEFAULT_ARR_CUTOFF = ["0-10"]
RUN_LOG = Path("pt_file_run.log").resolve()
FAIL_LOG = Path("pt_file_failures.log").resolve()


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints full help text when an error occurs."""

    def error(self, message: str) -> None:  # pragma: no cover
        self.print_help(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def _trim_suffix(stem: str, suffixes: tuple[str, ...]) -> str:
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem = stem.rstrip("_- .")
            lower = stem.lower()
    return stem


def _normalise_interface_name(name: str) -> str:
    return _trim_suffix(Path(name).stem, (".interface", "interface", "iface"))


def _normalise_topology_name(name: str) -> str:
    return _trim_suffix(Path(name).stem, (".topology", "topology"))


def _normalise_node_name(name: str) -> str:
    return _trim_suffix(Path(name).stem, (".node_fea", "node_fea", "node"))


def _gather_files(root: Path, patterns: Iterable[str], normalise) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                key = normalise(path.name)
                mapping.setdefault(key, path)
    return mapping


def _index_structures(dataset_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for ext in (".pdb",):
        for path in dataset_dir.rglob(f"*{ext}"):
            if path.is_file():
                mapping.setdefault(path.stem, path)
    return mapping


def parse_args(argv: Optional[Iterable[str]] = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = HelpOnErrorArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--interface-dir", type=Path, required=True,
                        help="Directory containing interface files (.interface.txt). (required)")
    parser.add_argument("--topology-dir", type=Path, required=True,
                        help="Directory containing topology CSV files (.topology.csv). (required)")
    parser.add_argument("--node-dir", type=Path, required=True,
                        help="Directory containing node feature CSV files (.csv). (required)")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Directory containing source structures (.pdb). (required)")
    parser.add_argument("--output-pt-dir", type=Path, required=True,
                        help="Directory where .pt graphs will be written. (required)")
    parser.add_argument("--arr-cutoff", nargs="*", default=DEFAULT_ARR_CUTOFF,
                        help="Edge distance cutoffs passed to create_graph (default: %(default)s)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of worker threads (1 = sequential).")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.arr_cutoff:
        args.arr_cutoff = DEFAULT_ARR_CUTOFF
    return args, parser


@dataclass
class GraphTask:
    model: str
    pdb_path: Path
    interface_path: Path
    topology_path: Path
    node_path: Path


def _prepare_staging(task: GraphTask) -> tuple[tempfile.TemporaryDirectory, Path, Path, Path, Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix=f"pt_build_{task.model}_")
    root = Path(temp_dir.name)
    node_dir = root / "node"
    interface_dir = root / "interface"
    topology_dir = root / "topology"
    pdb_dir = root / "pdb"
    node_dir.mkdir(parents=True, exist_ok=True)
    interface_dir.mkdir(parents=True, exist_ok=True)
    topology_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task.node_path, node_dir / f"{task.model}.csv")
    shutil.copyfile(task.interface_path, interface_dir / f"{task.model}.txt")
    shutil.copyfile(task.topology_path, topology_dir / f"{task.model}.topology.csv")
    shutil.copyfile(task.pdb_path, pdb_dir / f"{task.model}{task.pdb_path.suffix}")
    return temp_dir, node_dir, interface_dir, topology_dir, pdb_dir


def _process_task(task: GraphTask, output_dir: Path, arr_cutoff: List[str]) -> tuple[str, Optional[str]]:
    temp_handle: Optional[tempfile.TemporaryDirectory] = None
    try:
        temp_handle, node_dir, interface_dir, topology_dir, pdb_dir = _prepare_staging(task)
        create_graph(
            task.model,
            str(node_dir),
            str(interface_dir),
            arr_cutoff,
            str(output_dir),
            str(pdb_dir),
        )
        return task.model, None
    except Exception as exc:  # pragma: no cover
        return task.model, str(exc)
    finally:
        if temp_handle is not None:
            temp_handle.cleanup()


def _log_configuration(log, args: argparse.Namespace, defaults: Dict[str, object]) -> None:
    log("=== write_pt_file run ===")
    log(f"Start time: {dt.datetime.now().strftime('%H:%M:%S')}")
    log("Configuration:")
    for key in sorted(vars(args)):
        log(f"  {key}: {getattr(args, key)}")
    log("Defaults:")
    for key in sorted(defaults):
        log(f"  {key}: {defaults[key]}")


def _collect_defaults(parser: argparse.ArgumentParser) -> Dict[str, object]:
    defaults: Dict[str, object] = {}
    for action in parser._actions:
        if action.dest and action.default is not argparse.SUPPRESS:
            defaults[action.dest] = action.default
    return defaults


def main(argv: Optional[Iterable[str]] = None) -> int:
    args, parser = parse_args(argv)
    defaults = _collect_defaults(parser)

    output_dir = args.output_pt_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    with RUN_LOG.open("w", encoding="utf-8") as run_handle, FAIL_LOG.open("w", encoding="utf-8") as fail_handle:
        def log(msg: str) -> None:
            print(msg)
            run_handle.write(msg + "\n")

        _log_configuration(log, args, defaults)

        interface_dir = args.interface_dir.resolve()
        topo_dir = args.topology_dir.resolve()
        node_dir = args.node_dir.resolve()
        dataset_dir = args.dataset_dir.resolve()

        for path, label in (
            (interface_dir, "interface directory"),
            (topo_dir, "topology directory"),
            (node_dir, "node directory"),
            (dataset_dir, "dataset directory"),
        ):
            if not path.is_dir():
                message = f"Error: {label} not found: {path}"
                log(message)
                fail_handle.write(message + "\n")
                return 2

        interface_map = _gather_files(interface_dir, ("*.interface.txt", "*.txt"), _normalise_interface_name)
        topology_map = _gather_files(topo_dir, ("*.topology.csv", "*.csv"), _normalise_topology_name)
        node_map = _gather_files(node_dir, ("*.csv",), _normalise_node_name)
        structure_map = _index_structures(dataset_dir)

        shared_models = sorted(set(interface_map) & set(topology_map) & set(node_map) & set(structure_map))
        if not shared_models:
            message = "[warn] No common models across interface/topology/node/dataset directories"
            log(message)
            fail_handle.write(message + "\n")
            return 1

        tasks: List[GraphTask] = []
        for model in shared_models:
            tasks.append(
                GraphTask(
                    model=model,
                    pdb_path=structure_map[model],
                    interface_path=interface_map[model],
                    topology_path=topology_map[model],
                    node_path=node_map[model],
                )
            )

        failures: List[tuple[str, str]] = []
        success = 0

        if args.jobs and args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                future_map = {executor.submit(_process_task, task, output_dir, args.arr_cutoff): task for task in tasks}
                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        model, error = future.result()
                    except Exception as exc:  # pragma: no cover
                        error = str(exc)
                        failures.append((task.model, error))
                        fail_handle.write(
                            f"Model: {task.model}\n"
                            f"  interface: {task.interface_path}\n"
                            f"  topology: {task.topology_path}\n"
                            f"  node: {task.node_path}\n"
                            f"  pdb: {task.pdb_path}\n"
                            f"  error: {error}\n\n"
                        )
                        log(f"[warn] {task.model}: {error}")
                    else:
                        if error:
                            failures.append((model, error))
                            fail_handle.write(
                                f"Model: {model}\n"
                                f"  interface: {task.interface_path}\n"
                                f"  topology: {task.topology_path}\n"
                                f"  node: {task.node_path}\n"
                                f"  pdb: {task.pdb_path}\n"
                                f"  error: {error}\n\n"
                            )
                            log(f"[warn] {model}: {error}")
                        else:
                            success += 1
        else:
            for task in tasks:
                model, error = _process_task(task, output_dir, args.arr_cutoff)
                if error:
                    failures.append((model, error))
                    fail_handle.write(
                        f"Model: {model}\n"
                        f"  interface: {task.interface_path}\n"
                        f"  topology: {task.topology_path}\n"
                        f"  node: {task.node_path}\n"
                        f"  pdb: {task.pdb_path}\n"
                        f"  error: {error}\n\n"
                    )
                    log(f"[warn] {model}: {error}")
                else:
                    success += 1

        elapsed = time.perf_counter() - start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        log("Summary:")
        log(f"  processed: {len(tasks)}")
        log(f"  succeeded: {success}")
        log(f"  failed:    {len(failures)}")
        log(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        if failures:
            log("Failures recorded in pt_file_failures.log")
            return 1

        log("All .pt files generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
