from __future__ import annotations

import inspect
import hashlib
import json
import os
import time
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

from .node_features import node_fea
from .node_utils import canonical_id_order
from .progress import StageProgress
from .stage_common import (
    gather_files,
    normalise_interface_name,
    normalise_topology_name,
    normalise_node_name,
    relative_key,
)


def _hash_array(arr) -> str:
    """
    Hash a numeric view in a way that mirrors what flows into the model:
    - Preserve canonical row order on ID if present
    - Cast to float32 to discard sub-ULP float64 noise across platforms
    """
    if isinstance(arr, pd.DataFrame):
        view = arr.copy()
    else:
        view = pd.DataFrame(arr)
    if "ID" in view.columns:
        view = view.sort_values("ID").reset_index(drop=True)
    try:
        numeric = view.to_numpy(dtype=np.float32, copy=True)
    except Exception:
        # Fall back to the raw bytes if coercion fails for any reason
        numeric = view.to_numpy(copy=True)
    return hashlib.sha256(numeric.tobytes()).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats_frame(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return {"rows": len(df), "cols": len(df.columns)}
    values = numeric.to_numpy()
    return {
        "rows": int(values.shape[0]),
        "cols": int(values.shape[1]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


class NodeTracer:
    FLAG = "QTOPO_NODE_TRACE"
    FILTER = "QTOPO_NODE_TRACE_FILTER"
    DIR = "QTOPO_NODE_TRACE_DIR"

    def __init__(self, enabled: bool, trace_dir: Path, filters: tuple[str, ...]):
        self.enabled = enabled
        self.trace_dir = trace_dir
        self.filters = filters

    @classmethod
    def from_env(cls, model_key: str) -> "NodeTracer":
        enabled = os.environ.get(cls.FLAG, "").lower() in {"1", "true", "yes"}
        trace_dir = Path(os.environ.get(cls.DIR, "./node_trace")).expanduser()
        raw = os.environ.get(cls.FILTER, "")
        filters = tuple(token.strip() for token in raw.split(",") if token.strip())
        if filters and not any(token in model_key for token in filters):
            enabled = False
        return cls(enabled, trace_dir, filters)

    def record(self, model_key: str, stage: str, payload: dict) -> None:
        if not self.enabled:
            return
        try:
            target = self.trace_dir / model_key / f"{stage}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            warnings.warn(f"Node tracing failed for {model_key}:{stage}", RuntimeWarning)


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


def _process_task(task: NodeTask, drop_na: bool, sort_artifacts: bool) -> Tuple[str, Optional[str]]:
    log_lines: List[str] = []
    temp: Optional[tempfile.TemporaryDirectory] = None
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    tracer = NodeTracer.from_env(task.model_key)
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
        if tracer.enabled:
            tracer.record(
                task.model_key,
                "inputs",
                {
                    "interface_hash": _hash_file(task.interface_path),
                    "topology_hash": _hash_file(task.topology_path),
                },
            )
            tracer.record(
                task.model_key,
                "pre_drop",
                {
                    "hash": _hash_array(fea_df),
                    "stats": _stats_frame(fea_df),
                    "columns": list(fea_df.columns),
                },
            )
        if drop_na:
            pd.set_option("future.no_silent_downcasting", True)
            fea_df.replace("NA", pd.NA, inplace=True)
            fea_df = fea_df.dropna()
        fea_df = _canonicalise_node_df(fea_df, sort_artifacts=sort_artifacts)
        if tracer.enabled:
            tracer.record(
                task.model_key,
                "post_drop",
                {
                    "hash": _hash_array(fea_df),
                    "stats": _stats_frame(fea_df),
                    "columns": list(fea_df.columns),
                },
            )
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
    sort_artifacts: bool = True,
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

    progress = StageProgress("Node", len(node_tasks), dataset_name=dataset_dir.name)

    if node_tasks:
        start = time.perf_counter()
        worker_count = max(1, int(jobs)) if jobs else 1
        if worker_count <= 1:
            for task in node_tasks:
                model_key, error = _process_task(task, drop_na, sort_artifacts)
                if error:
                    failures.append((model_key, task.log_path, error))
                else:
                    success += 1
                progress.increment()
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(_process_task, task, drop_na, sort_artifacts): task.model_key
                    for task in node_tasks
                }
                for future in as_completed(future_map):
                    model_key, error = future.result()
                    if error:
                        failures.append((model_key, task_map[model_key].log_path, error))
                    else:
                        success += 1
                    progress.increment()
        elapsed = time.perf_counter() - start

    return {
        "output_dir": node_dir,
        "log_dir": node_log_dir,
        "success": success,
        "failures": failures,
        "elapsed": elapsed,
        "processed": len(node_tasks),
    }


def _canonicalise_node_df(fea_df: pd.DataFrame, sort_artifacts: bool = True) -> pd.DataFrame:
    if not sort_artifacts or "ID" not in fea_df.columns:
        return fea_df.reset_index(drop=True)
    order = canonical_id_order(list(fea_df["ID"]))
    if order == list(range(len(order))):
        return fea_df.reset_index(drop=True)
    return fea_df.iloc[order].reset_index(drop=True)
