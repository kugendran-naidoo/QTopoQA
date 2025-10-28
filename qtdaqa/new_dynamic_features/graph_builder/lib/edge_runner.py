from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from dataclasses import dataclass

from .stage_common import (
    gather_files,
    normalise_interface_name,
    normalise_topology_name,
    normalise_node_name,
    index_structures,
)
from .edge_common import InterfaceResidue, StructureCache

try:  # support execution as package or script
    from ..modules.base import EdgeFeatureModule, EdgeBuildResult
except ImportError:  # pragma: no cover
    from modules.base import EdgeFeatureModule, EdgeBuildResult  # type: ignore

LOG = logging.getLogger(__name__)


@dataclass
class GraphTask:
    model_key: str
    model_name: str
    pdb_path: Path
    interface_path: Path
    topology_path: Path
    node_path: Path
    log_path: Path


def _parse_interface_file(path: Path) -> List[InterfaceResidue]:
    residues: List[InterfaceResidue] = []
    import re

    pattern = re.compile(
        r"^c<(?P<chain>[^>]+)>r<(?P<resnum>-?\d+)>(?:i<(?P<icode>[^>]+)>)?R<(?P<resname>[^>]+)>\s+"
        r"(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+(?P<z>-?\d+(?:\.\d+)?)"
    )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            match = pattern.match(stripped)
            if not match:
                continue
            coord = np.array(
                [
                    float(match.group("x")),
                    float(match.group("y")),
                    float(match.group("z")),
                ],
                dtype=float,
            )
            residues.append(
                InterfaceResidue(
                    descriptor=stripped.split()[0],
                    chain_id=match.group("chain"),
                    residue_seq=int(match.group("resnum")),
                    insertion_code=(match.group("icode") or " ").strip(),
                    residue_name=match.group("resname"),
                    coord=coord,
                )
            )
    return residues


def _load_node_features(node_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(node_path)
    if "ID" not in df.columns:
        raise ValueError(f"Node feature file {node_path} missing 'ID' column.")
    feature_cols = [col for col in df.columns if col != "ID"]
    return df, feature_cols


def _prepare_tasks(
    interface_dir: Path,
    topology_dir: Path,
    node_dir: Path,
    dataset_dir: Path,
    log_dir: Path,
) -> Tuple[List[GraphTask], Dict[str, Path]]:
    interface_map = gather_files(interface_dir, ("*.interface.txt", "*.txt"), normalise_interface_name)
    topology_map = gather_files(topology_dir, ("*.topology.csv", "*.csv"), normalise_topology_name)
    node_map = gather_files(node_dir, ("*.csv",), normalise_node_name)
    structure_map = index_structures(dataset_dir, (".pdb",))
    shared_models = sorted(set(interface_map) & set(topology_map) & set(node_map) & set(structure_map))
    model_log_dir = log_dir / "models"
    model_log_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[GraphTask] = []
    for model_key in shared_models:
        model_name = Path(model_key).name
        log_rel = Path(model_key).with_suffix(".log")
        log_path = (model_log_dir / log_rel).resolve()
        tasks.append(
            GraphTask(
                model_key=model_key,
                model_name=model_name,
                pdb_path=structure_map[model_key],
                interface_path=interface_map[model_key],
                topology_path=topology_map[model_key],
                node_path=node_map[model_key],
                log_path=log_path,
            )
        )
    return tasks, structure_map


def run_edge_stage(
    dataset_dir: Path,
    interface_dir: Path,
    topology_dir: Path,
    node_dir: Path,
    output_dir: Path,
    log_dir: Path,
    edge_module: EdgeFeatureModule,
    jobs: Optional[int] = None,
    edge_dump_dir: Optional[Path] = None,
) -> Dict[str, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if edge_dump_dir is not None:
        edge_dump_dir.mkdir(parents=True, exist_ok=True)

    tasks, structure_map = _prepare_tasks(interface_dir, topology_dir, node_dir, dataset_dir, log_dir)
    run_log_path = log_dir / "edge_stage.log"
    logger = logging.getLogger("edge_stage")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    success = 0
    failures: List[Tuple[str, str, Path]] = []
    edge_dim: Optional[int] = None
    metadata_records: Dict[str, Dict[str, object]] = {}

    worker_count = max(1, int(jobs)) if jobs else 1

    start = time.perf_counter()

    if worker_count <= 1:
        for task in tasks:
            result = _process_task(task, edge_module, output_dir, edge_dump_dir, metadata_records)
            if result["error"]:
                failures.append((task.model_key, result["error"], task.log_path))
            else:
                success += 1
                edge_dim = result["edge_dim"]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(_process_task, task, edge_module, output_dir, edge_dump_dir, metadata_records): task
                for task in tasks
            }
            for future, task in future_map.items():
                result = future.result()
                if result["error"]:
                    failures.append((task.model_key, result["error"], task.log_path))
                else:
                    success += 1
                    edge_dim = result["edge_dim"]

    metadata_path = output_dir / "graph_metadata.json"
    metadata_path.write_text(json.dumps(metadata_records, indent=2, sort_keys=True), encoding="utf-8")

    elapsed = time.perf_counter() - start

    return {
        "processed": len(tasks),
        "success": success,
        "failures": failures,
        "run_log": run_log_path,
        "log_dir": log_dir,
        "output_dir": output_dir,
        "edge_feature_dim": edge_dim,
        "metadata_path": metadata_path,
        "structure_map": structure_map,
        "elapsed": elapsed,
    }


def _process_task(
    task: GraphTask,
    edge_module: EdgeFeatureModule,
    output_dir: Path,
    edge_dump_dir: Optional[Path],
    metadata_records: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    log_lines: List[str] = []
    try:
        node_df, feature_cols = _load_node_features(task.node_path)
        residues = _parse_interface_file(task.interface_path)
        if not residues:
            raise RuntimeError(f"No interface residues parsed for {task.model_key}")
        id_to_index = {identifier: idx for idx, identifier in enumerate(node_df["ID"].tolist())}
        structure_cache = StructureCache(task.pdb_path)

        dump_path = None
        if edge_dump_dir is not None:
            dump_path = (edge_dump_dir / Path(task.model_key).with_suffix(".edges.csv")).resolve()
            dump_path.parent.mkdir(parents=True, exist_ok=True)

        edge_result: EdgeBuildResult = edge_module.build_edges(
            model_key=task.model_key,
            residues=residues,
            id_to_index=id_to_index,
            structure=structure_cache,
            node_df=node_df,
            interface_path=task.interface_path,
            topology_path=task.topology_path,
            node_path=task.node_path,
            pdb_path=task.pdb_path,
            dump_path=dump_path,
        )

        x = node_df[feature_cols].to_numpy(dtype=np.float32)
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_result.edge_index.T if edge_result.edge_index.size else np.empty((2, 0), dtype=np.int64), dtype=torch.long),
            edge_attr=torch.tensor(edge_result.edge_attr, dtype=torch.float32),
        )
        data.metadata = {
            "edge_module": edge_module.metadata().module_id,
            "edge_params": edge_module.params,
            "edge_info": edge_result.metadata,
        }
        output_path = output_dir / Path(f"{task.model_key}.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        metadata_records[task.model_key] = {
            "edge_module": edge_module.metadata().module_id,
            "edge_params": edge_module.params,
            "edge_metadata": edge_result.metadata,
            "node_feature_columns": feature_cols,
        }
        log_lines.extend(
            [
                f"Model key: {task.model_key}",
                "Status: SUCCESS",
                f"PDB: {task.pdb_path}",
                f"Interface: {task.interface_path}",
                f"Topology: {task.topology_path}",
                f"Node features: {task.node_path}",
                f"Output pt: {output_path}",
                f"Edge dim: {edge_result.edge_attr.shape[1] if edge_result.edge_attr.size else 0}",
            ]
        )
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {"error": None, "edge_dim": edge_result.edge_attr.shape[1] if edge_result.edge_attr.size else 0}
    except Exception as exc:  # pragma: no cover
        log_lines.extend(
            [
                f"Model key: {task.model_key}",
                "Status: FAILURE",
                f"Error: {exc}",
            ]
        )
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        task.log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {"error": str(exc), "edge_dim": None}
