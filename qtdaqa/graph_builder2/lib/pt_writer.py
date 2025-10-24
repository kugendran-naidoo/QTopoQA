"""Library helpers for building graph .pt files from precomputed features."""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required to generate .pt graph files.") from exc

DEFAULT_ARR_CUTOFF: Sequence[str] = ("0-10",)


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


@dataclass
class GraphTask:
    model: str
    pdb_path: Path
    interface_path: Path
    topology_path: Path
    node_path: Path
    log_path: Path


@dataclass
class PtGenerationResult:
    processed: int
    success_count: int
    failures: List[tuple[str, str, Path]]
    run_log: Path
    log_dir: Path


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


def _process_task(
    task: GraphTask,
    output_dir: Path,
    arr_cutoff: Sequence[str],
    dump_edges: bool,
    edge_dump_dir: Optional[Path],
) -> tuple[str, Optional[str]]:
    temp_handle: Optional[tempfile.TemporaryDirectory] = None
    try:
        temp_handle, node_dir, interface_dir, topology_dir, pdb_dir = _prepare_staging(task)
        create_graph(
            task.model,
            str(node_dir),
            str(interface_dir),
            list(arr_cutoff),
            str(output_dir),
            str(pdb_dir),
            dump_edges=dump_edges,
            edge_dir=str(edge_dump_dir) if (dump_edges and edge_dump_dir is not None) else None,
        )
        return task.model, None
    except Exception as exc:  # pragma: no cover
        return task.model, str(exc)
    finally:
        if temp_handle is not None:
            temp_handle.cleanup()


def generate_pt_files(
    interface_dir: Path,
    topology_dir: Path,
    node_dir: Path,
    dataset_dir: Path,
    output_pt_dir: Path,
    jobs: int = 1,
    arr_cutoff: Optional[Sequence[str]] = None,
    log_dir: Optional[Path] = None,
    logger: Optional["logging.Logger"] = None,
    dump_edges: bool = True,
    edge_dump_dir: Optional[Path] = None,
) -> PtGenerationResult:
    import logging

    arr_cutoff = tuple(arr_cutoff) if arr_cutoff else DEFAULT_ARR_CUTOFF

    interface_dir = interface_dir.resolve()
    topology_dir = topology_dir.resolve()
    node_dir = node_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    output_pt_dir = output_pt_dir.resolve()
    output_pt_dir.mkdir(parents=True, exist_ok=True)
    if edge_dump_dir is not None:
        edge_dump_dir = edge_dump_dir.resolve()
        edge_dump_dir.mkdir(parents=True, exist_ok=True)

    if log_dir is None:
        log_dir = Path("pt_logs")
    log_dir = log_dir.resolve()
    model_log_dir = log_dir / "models"
    model_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / "pt_run.log"

    interface_map = _gather_files(interface_dir, ("*.interface.txt", "*.txt"), _normalise_interface_name)
    topology_map = _gather_files(topology_dir, ("*.topology.csv", "*.csv"), _normalise_topology_name)
    node_map = _gather_files(node_dir, ("*.csv",), _normalise_node_name)
    structure_map = _index_structures(dataset_dir)

    shared_models = sorted(set(interface_map) & set(topology_map) & set(node_map) & set(structure_map))
    tasks: List[GraphTask] = []
    for model in shared_models:
        log_path = model_log_dir / f"{model}.log"
        tasks.append(
            GraphTask(
                model=model,
                pdb_path=structure_map[model],
                interface_path=interface_map[model],
                topology_path=topology_map[model],
                node_path=node_map[model],
                log_path=log_path,
            )
        )

    missing_interface = sorted(set(node_map) - set(interface_map))
    missing_topology = sorted(set(node_map) - set(topology_map))
    missing_structure = sorted(set(node_map) - set(structure_map))

    logger = logger or logging.getLogger(__name__)
    logger.info("Preparing .pt graph generation (%d models intersected)", len(shared_models))
    if missing_interface:
        logger.warning("Skipping %d models with missing interface files", len(missing_interface))
    if missing_topology:
        logger.warning("Skipping %d models with missing topology files", len(missing_topology))
    if missing_structure:
        logger.warning("Skipping %d models with missing PDB structures", len(missing_structure))

    start = time.perf_counter()
    success = 0
    failures: List[tuple[str, str, Path]] = []

    def _write_model_log(task: GraphTask, status: str, message: str = "") -> None:
        lines = [
            f"PDB: {task.pdb_path}",
            f"Interface: {task.interface_path}",
            f"Topology: {task.topology_path}",
            f"Node features: {task.node_path}",
            f"Output: {output_pt_dir / f'{task.model}.pt'}",
            f"Status: {status}",
        ]
        if message:
            lines.append(f"Message: {message}")
        task.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if tasks:
        if jobs <= 1:
            for task in tasks:
                model, error = _process_task(task, output_pt_dir, arr_cutoff, dump_edges, edge_dump_dir)
                if error:
                    failures.append((model, error, task.log_path))
                    _write_model_log(task, "FAILURE", error)
                else:
                    success += 1
                    _write_model_log(task, "SUCCESS")
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {
                    executor.submit(_process_task, task, output_pt_dir, arr_cutoff, dump_edges, edge_dump_dir): task
                    for task in tasks
                }
                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        model, error = future.result()
                    except Exception as exc:  # pragma: no cover
                        error = str(exc)
                        model = task.model
                    if error:
                        failures.append((model, error, task.log_path))
                        _write_model_log(task, "FAILURE", error)
                    else:
                        success += 1
                        _write_model_log(task, "SUCCESS")
    elapsed = time.perf_counter() - start
    logger.info(".pt generation completed: %d success, %d failure (%.2f s)", success, len(failures), elapsed)

    run_lines: List[str] = []
    run_lines.append("=== pt generation run ===")
    run_lines.append(f"Start time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run_lines.append(f"Interface dir : {interface_dir}")
    run_lines.append(f"Topology dir  : {topology_dir}")
    run_lines.append(f"Node dir      : {node_dir}")
    run_lines.append(f"PDB dir       : {dataset_dir}")
    run_lines.append(f"Output dir    : {output_pt_dir}")
    run_lines.append(f"Workers       : {jobs}")
    run_lines.append(f"Cutoffs       : {list(arr_cutoff)}")
    run_lines.append("")
    run_lines.append(f"Total candidates : {len(tasks)}")
    run_lines.append(f"Successes        : {success}")
    run_lines.append(f"Failures         : {len(failures)}")
    run_lines.append(f"Elapsed          : {elapsed:0.2f} s")
    if missing_interface or missing_topology or missing_structure:
        run_lines.append("")
        run_lines.append("Missing resources:")
        if missing_interface:
            run_lines.append(f"  Interface missing: {len(missing_interface)}")
        if missing_topology:
            run_lines.append(f"  Topology missing : {len(missing_topology)}")
        if missing_structure:
            run_lines.append(f"  Structures missing: {len(missing_structure)}")

    run_log_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    return PtGenerationResult(
        processed=len(tasks),
        success_count=success,
        failures=failures,
        run_log=run_log_path,
        log_dir=log_dir,
    )
