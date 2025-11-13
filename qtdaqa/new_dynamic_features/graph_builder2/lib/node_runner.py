from __future__ import annotations

import inspect
import time
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .node_features import node_fea
from .node_utils import canonical_id_order
from .stage_common import (
    gather_files,
    normalise_interface_name,
    normalise_topology_name,
    normalise_node_name,
    relative_key,
)


@dataclass
class NodeTask:
    model_key: str
    model_name: str
    structure_path: Path
    interface_path: Path
    topology_path: Path
    output_path: Path
    log_path: Path


def _detect_node_fea_signature() -> int:
    try:
        return len(inspect.signature(node_fea).parameters)
    except (TypeError, ValueError):
        return -1


def _initialise_node_fea(task: NodeTask, iface_dir: Path, topo_dir: Path):
    signature_size = _detect_node_fea_signature()
    iface = str(iface_dir)
    topo = str(topo_dir)
    if signature_size == 3:
        return node_fea(str(task.structure_path), iface, topo)
    if signature_size == 4:
        return node_fea(task.model_name, str(task.structure_path.parent), iface, topo)
    try:
        return node_fea(str(task.structure_path), iface, topo)
    except TypeError:
        return node_fea(task.model_name, str(task.structure_path.parent), iface, topo)


def _stage_inputs(task: NodeTask) -> Tuple[tempfile.TemporaryDirectory, Path, Path]:
    safe_token = task.model_name.replace("/", "_")
    temp_dir = tempfile.TemporaryDirectory(prefix=f"node_features_{safe_token}_")
    root = Path(temp_dir.name)
    iface_dir = root / "interface"
    topo_dir = root / "topology"
    iface_dir.mkdir(parents=True, exist_ok=True)
    topo_dir.mkdir(parents=True, exist_ok=True)
    iface_target = iface_dir / f"{task.model_name}.txt"
    topo_target = topo_dir / f"{task.model_name}.csv"
    iface_target.parent.mkdir(parents=True, exist_ok=True)
    topo_target.parent.mkdir(parents=True, exist_ok=True)
    iface_target.write_bytes(task.interface_path.read_bytes())
    topo_target.write_bytes(task.topology_path.read_bytes())
    return temp_dir, iface_dir, topo_dir


def _process_task(task: NodeTask, drop_na: bool) -> Tuple[str, Optional[str]]:
    log_lines: List[str] = []
    temp: Optional[tempfile.TemporaryDirectory] = None
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temp, iface_dir, topo_dir = _stage_inputs(task)
        extractor = _initialise_node_fea(task, iface_dir, topo_dir)
        log_lines.extend(
            [
                f"Model key: {task.model_key}",
                f"Model name: {task.model_name}",
                f"PDB: {task.structure_path}",
                f"Interface source: {task.interface_path}",
                f"Topology source: {task.topology_path}",
                f"Output CSV: {task.output_path}",
            ]
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = extractor.calculate_fea()
        if isinstance(result, tuple):
            fea_df = result[0]
        else:
            fea_df = result
        if drop_na:
            pd.set_option("future.no_silent_downcasting", True)
            fea_df.replace("NA", pd.NA, inplace=True)
            fea_df = fea_df.dropna()
        fea_df = _canonicalise_node_df(fea_df)
        fea_df.to_csv(task.output_path, index=False)
        if captured:
            log_lines.append("Warnings:")
            for warn in captured:
                log_lines.append(f"  {warn.category.__name__}: {warn.message}")
        log_lines.append("Status: SUCCESS")
        return task.model_key, None
    except Exception as exc:  # pragma: no cover
        log_lines.append("Status: FAILURE")
        log_lines.append(f"Error: {exc}")
        return task.model_key, str(exc)
    finally:
        if temp is not None:
            temp.cleanup()
        if log_lines:
            task.log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def run_node_stage(
    dataset_dir: Path,
    structure_map: Dict[str, Path],
    interface_dir: Path,
    topology_dir: Path,
    work_dir: Path,
    log_dir: Path,
    drop_na: bool = False,
    jobs: Optional[int] = None,
) -> Dict[str, object]:
    node_dir = work_dir / "node_features"
    node_dir.mkdir(parents=True, exist_ok=True)
    node_log_dir = log_dir / "node_feature_logs"
    node_log_dir.mkdir(parents=True, exist_ok=True)

    interface_map = gather_files(interface_dir, ("*.interface.txt", "*.txt"), normalise_interface_name)
    topology_map = gather_files(topology_dir, ("*.topology.csv", "*.csv"), normalise_topology_name)
    node_tasks: List[NodeTask] = []
    shared = sorted(set(structure_map) & set(interface_map) & set(topology_map))
    for key in shared:
        model_name = Path(key).name
        output_rel = Path(relative_key(interface_dir, interface_map[key], model_name)).with_suffix(".csv")
        output_path = (node_dir / output_rel).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_rel = output_rel.with_suffix(".log")
        log_path = (node_log_dir / log_rel).resolve()
        node_tasks.append(
            NodeTask(
                model_key=key,
                model_name=model_name,
                structure_path=structure_map[key],
                interface_path=interface_map[key],
                topology_path=topology_map[key],
                output_path=output_path,
                log_path=log_path,
            )
        )

    success = 0
    failures: List[Tuple[str, Path, str]] = []
    elapsed = 0.0
    task_map = {task.model_key: task for task in node_tasks}

    if node_tasks:
        start = time.perf_counter()
        worker_count = max(1, int(jobs)) if jobs else 1
        if worker_count <= 1:
            for task in node_tasks:
                model_key, error = _process_task(task, drop_na)
                if error:
                    failures.append((model_key, task.log_path, error))
                else:
                    success += 1
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {executor.submit(_process_task, task, drop_na): task.model_key for task in node_tasks}
                for future in as_completed(future_map):
                    model_key, error = future.result()
                    if error:
                        failures.append((model_key, task_map[model_key].log_path, error))
                    else:
                        success += 1
        elapsed = time.perf_counter() - start

    return {
        "output_dir": node_dir,
        "log_dir": node_log_dir,
        "success": success,
        "failures": failures,
        "elapsed": elapsed,
        "processed": len(node_tasks),
    }


def _canonicalise_node_df(fea_df: pd.DataFrame) -> pd.DataFrame:
    if "ID" not in fea_df.columns:
        return fea_df.reset_index(drop=True)
    order = canonical_id_order(list(fea_df["ID"]))
    if order == list(range(len(order))):
        return fea_df.reset_index(drop=True)
    return fea_df.iloc[order].reset_index(drop=True)
