"""Library helpers for building graph .pt files from precomputed features (multi-scale edges)."""
from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

try:  # pragma: no cover - support running as a script
    from .pdb_utils import create_pdb_parser
except ImportError:  # pragma: no cover
    from pdb_utils import create_pdb_parser  # type: ignore

try:  # when imported via package
    from .config import BuilderConfig, EdgeBand
except ImportError:  # when running as stand-alone script with lib on sys.path
    from config import BuilderConfig, EdgeBand


def _locate_repo_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        candidate = parent / "topoqa" / "src"
        if candidate.exists():
            return parent
    raise RuntimeError("Unable to locate repo root containing 'topoqa/src'.")


REPO_ROOT = _locate_repo_root(Path(__file__).resolve())
TOPOQA_DIR = REPO_ROOT / "topoqa"
TOPOQA_SRC = TOPOQA_DIR / "src"
if str(TOPOQA_SRC) not in sys.path:
    sys.path.insert(0, str(TOPOQA_SRC))


CHAIN_BASE = ord("A")
AMINO_ACIDS = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]
AA_INDEX = {name: idx for idx, name in enumerate(AMINO_ACIDS)}


@dataclass
class GraphTask:
    model_key: str
    model_name: str
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
    edge_feature_dim: Optional[int] = None


@dataclass
class InterfaceResidue:
    descriptor: str
    chain_id: str
    residue_seq: int
    insertion_code: str
    residue_name: str
    coord: np.ndarray


class StructureCache:
    def __init__(self, pdb_path: Path):
        parser = create_pdb_parser()
        self.structure = parser.get_structure("protein", str(pdb_path))

    def get_residue(self, chain_id: str, residue_seq: int, insertion_code: str):
        insertion = insertion_code or " "
        try:
            return self.structure[0][chain_id][(" ", residue_seq, insertion)]
        except KeyError:
            return None


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


def _relative_key(root: Path, path: Path, name: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return name
    parent_parts = [part for part in relative.parent.parts if part not in ("", ".")]
    if parent_parts:
        return str(PurePosixPath(*parent_parts, name))
    return name


def _structure_model_key(dataset_dir: Path, structure_path: Path) -> str:
    relative = structure_path.relative_to(dataset_dir)
    parent_parts = [part for part in relative.parent.parts if part not in ("", ".")]
    if parent_parts:
        return str(PurePosixPath(*parent_parts, structure_path.stem))
    return structure_path.stem


def _gather_files(root: Path, patterns: Iterable[str], normalise) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                normalised = normalise(path.name)
                key = _relative_key(root, path, normalised)
                mapping.setdefault(key, path)
    return mapping


def _index_structures(dataset_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for ext in (".pdb",):
        for path in dataset_dir.rglob(f"*{ext}"):
            if path.is_file():
                key = _structure_model_key(dataset_dir, path)
                mapping.setdefault(key, path)
    return mapping


def _prepare_staging(task: GraphTask) -> tuple[tempfile.TemporaryDirectory, Path, Path, Path, Path]:
    safe_token = re.sub(r"[^A-Za-z0-9_.-]", "_", task.model_name) or "model"
    temp_dir = tempfile.TemporaryDirectory(prefix=f"pt_build_{safe_token}_")
    root = Path(temp_dir.name)
    node_dir = root / "node"
    interface_dir = root / "interface"
    topology_dir = root / "topology"
    pdb_dir = root / "pdb"
    node_dir.mkdir(parents=True, exist_ok=True)
    interface_dir.mkdir(parents=True, exist_ok=True)
    topology_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    node_target = node_dir / f"{task.model_name}.csv"
    iface_target = interface_dir / f"{task.model_name}.txt"
    topo_target = topology_dir / f"{task.model_name}.topology.csv"
    pdb_target = pdb_dir / f"{task.model_name}{task.pdb_path.suffix}"
    node_target.parent.mkdir(parents=True, exist_ok=True)
    iface_target.parent.mkdir(parents=True, exist_ok=True)
    topo_target.parent.mkdir(parents=True, exist_ok=True)
    pdb_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task.node_path, node_target)
    shutil.copyfile(task.interface_path, iface_target)
    shutil.copyfile(task.topology_path, topo_target)
    shutil.copyfile(task.pdb_path, pdb_target)
    return temp_dir, node_dir, interface_dir, topology_dir, pdb_dir


def _parse_interface_file(path: Path) -> List[InterfaceResidue]:
    residues: List[InterfaceResidue] = []
    pattern = re.compile(
        r"^c<(?P<chain>[^>]+)>r<(?P<resnum>-?\d+)>(?:i<(?P<icode>[^>]+)>)?R<(?P<resname>[^>]+)>\s+(?P<x>-?\d+(?:\.\d+)?)\s+(?P<y>-?\d+(?:\.\d+)?)\s+(?P<z>-?\d+(?:\.\d+)?)$"
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


def _encode_chain(chain_id: str) -> float:
    if not chain_id:
        return 0.0
    value = ord(chain_id.upper()[0]) - CHAIN_BASE + 1
    return max(0.0, min(1.0, value / 26.0))


def _encode_residue(res_name: str) -> float:
    idx = AA_INDEX.get(res_name.upper(), -1)
    if idx < 0:
        return 1.0
    return (idx + 1) / (len(AMINO_ACIDS) + 1)


def _atom_coordinates(residue) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        coord = atom.get_coord()
        coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


def _band_index(distance: float, bands: Sequence[EdgeBand]) -> Optional[int]:
    for idx, band in enumerate(bands):
        if band.min_distance <= distance < band.max_distance:
            return idx
    return None


def _compute_histogram(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    if coords_a.size == 0 or coords_b.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    diffs = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
    hist, _ = np.histogram(distances, bins=bins)
    total = hist.sum()
    if total > 0:
        return hist.astype(float) / float(total)
    return hist.astype(float)


class EdgeFeatureBuilder:
    def __init__(self, edge_config):
        self.edge_config = edge_config
        if len(edge_config.histogram_bins) < 2:
            raise ValueError("edge.histogram_bins must define at least two cut points.")
        self.hist_bins = np.asarray(edge_config.histogram_bins, dtype=float)

    def build_edges(
        self,
        residues: List[InterfaceResidue],
        id_to_index: Dict[str, int],
        structure: StructureCache,
        edge_dump_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        band_count = len(self.edge_config.bands)
        feature_rows: List[List[float]] = []
        edge_index: List[Tuple[int, int]] = []
        dump_rows: List[Dict[str, float]] = []

        for i, src in enumerate(residues):
            src_idx = id_to_index.get(src.descriptor)
            if src_idx is None:
                continue
            for j, dst in enumerate(residues):
                if i == j:
                    continue
                if src.chain_id == dst.chain_id:
                    continue
                dst_idx = id_to_index.get(dst.descriptor)
                if dst_idx is None:
                    continue
                vector = dst.coord - src.coord
                distance = float(np.linalg.norm(vector))
                band_idx = _band_index(distance, self.edge_config.bands)
                if band_idx is None:
                    continue
                features_forward = self._assemble_features(
                    distance,
                    vector,
                    src,
                    dst,
                    band_idx,
                    structure,
                )
                features_reverse = self._assemble_features(
                    distance,
                    -vector,
                    dst,
                    src,
                    band_idx,
                    structure,
                )
                feature_rows.append(features_forward)
                feature_rows.append(features_reverse)
                edge_index.append((src_idx, dst_idx))
                edge_index.append((dst_idx, src_idx))

                if edge_dump_path is not None:
                    dump_rows.append(
                        {
                            "src_idx": src_idx,
                            "dst_idx": dst_idx,
                            "src_id": src.descriptor,
                            "dst_id": dst.descriptor,
                            "band": self.edge_config.bands[band_idx].label,
                            "distance": distance,
                        }
                    )

        if not feature_rows:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 0), dtype=np.float32)

        feature_matrix = np.asarray(feature_rows, dtype=np.float32)
        edge_array = np.asarray(edge_index, dtype=np.int64)

        if edge_dump_path is not None:
            dump_df = pd.DataFrame(dump_rows)
            dump_df.to_csv(edge_dump_path, index=False)

        return edge_array, feature_matrix

    def _assemble_features(
        self,
        distance: float,
        vector: np.ndarray,
        src: InterfaceResidue,
        dst: InterfaceResidue,
        band_idx: int,
        structure: StructureCache,
    ) -> List[float]:
        eps = self.edge_config.unit_vector_epsilon
        features: List[float] = [distance]
        if self.edge_config.include_inverse_distance:
            features.append(1.0 / max(distance, eps))
        if self.edge_config.include_unit_vector:
            if distance > eps:
                unit_vec = vector / distance
            else:
                unit_vec = np.zeros(3, dtype=float)
            features.extend(unit_vec.tolist())

        features.append(_encode_chain(src.chain_id))
        features.append(_encode_chain(dst.chain_id))
        features.append(_encode_residue(src.residue_name))
        features.append(_encode_residue(dst.residue_name))
        features.append(float(band_idx))

        band_one_hot = [1.0 if idx == band_idx else 0.0 for idx in range(len(self.edge_config.bands))]
        features.extend(band_one_hot)

        residue_a = structure.get_residue(src.chain_id, src.residue_seq, src.insertion_code)
        residue_b = structure.get_residue(dst.chain_id, dst.residue_seq, dst.insertion_code)
        coords_a = _atom_coordinates(residue_a) if residue_a is not None else np.empty((0, 3))
        coords_b = _atom_coordinates(residue_b) if residue_b is not None else np.empty((0, 3))
        hist = _compute_histogram(coords_a, coords_b, self.hist_bins)
        features.extend(hist.tolist())

        if coords_a.size and coords_b.size:
            diffs = coords_a[:, None, :] - coords_b[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2)).reshape(-1)
            contact_count = np.count_nonzero(distances <= self.edge_config.contact_threshold)
            features.append(float(contact_count))
        else:
            features.append(0.0)

        return features


def _load_node_features(node_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(node_path)
    if "ID" not in df.columns:
        raise ValueError(f"Node feature file {node_path} missing 'ID' column.")
    feature_cols = [col for col in df.columns if col != "ID"]
    return df, feature_cols


def _save_graph(model: str, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, output_dir: Path) -> None:
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index.T if edge_index.size else np.empty((2, 0), dtype=np.int64), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
    )
    output_path = output_dir / Path(f"{model}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)


def _process_task(
    task: GraphTask,
    output_dir: Path,
    edge_dump_dir: Optional[Path],
    builder_config: BuilderConfig,
) -> tuple[str, Optional[str], Optional[int]]:
    temp_handle: Optional[tempfile.TemporaryDirectory] = None
    try:
        temp_handle, node_dir, interface_dir, topology_dir, pdb_dir = _prepare_staging(task)
        node_csv = node_dir / f"{task.model_name}.csv"
        interface_file = interface_dir / f"{task.model_name}.txt"
        pdb_file = next(pdb_dir.glob(f"{task.model_name}.*"))

        node_df, feature_cols = _load_node_features(node_csv)
        residues = _parse_interface_file(interface_file)
        if not residues:
            raise RuntimeError(f"No interface residues parsed for {task.model_key}")

        id_to_index = {identifier: idx for idx, identifier in enumerate(node_df["ID"].tolist())}
        structure_cache = StructureCache(pdb_file)
        feature_builder = EdgeFeatureBuilder(builder_config.edge)

        dump_path = None
        if edge_dump_dir is not None:
            edge_dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = edge_dump_dir / Path(f"{task.model_key}.edges.csv")
            dump_path.parent.mkdir(parents=True, exist_ok=True)

        edge_index, edge_attr = feature_builder.build_edges(residues, id_to_index, structure_cache, dump_path)
        x = node_df[feature_cols].to_numpy(dtype=np.float32)
        _save_graph(task.model_key, x, edge_index, edge_attr, output_dir)
        feature_dim = edge_attr.shape[1] if edge_attr.size else 0
        return task.model_key, None, feature_dim
    except Exception as exc:  # pragma: no cover
        return task.model_key, str(exc), None
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
    log_dir: Optional[Path] = None,
    logger: Optional["logging.Logger"] = None,
    edge_dump_dir: Optional[Path] = None,
    builder_config: Optional[BuilderConfig] = None,
) -> PtGenerationResult:
    import logging

    if builder_config is None:
        raise ValueError("builder_config must be provided.")

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
    feature_dim: Optional[int] = None

    def _write_model_log(task: GraphTask, status: str, message: str = "") -> None:
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"Model key: {task.model_key}",
            f"Model name: {task.model_name}",
            f"PDB: {task.pdb_path}",
            f"Interface: {task.interface_path}",
            f"Topology: {task.topology_path}",
            f"Node features: {task.node_path}",
            f"Output: {output_pt_dir / Path(f'{task.model_key}.pt')}",
            f"Status: {status}",
        ]
        if message:
            lines.append(f"Message: {message}")
        task.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if tasks:
        if jobs <= 1:
            for task in tasks:
                model, error, dim = _process_task(task, output_pt_dir, edge_dump_dir, builder_config)
                if error:
                    failures.append((model, error, task.log_path))
                    _write_model_log(task, "FAILURE", error)
                else:
                    success += 1
                    feature_dim = feature_dim or dim
                    _write_model_log(task, "SUCCESS")
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {
                    executor.submit(_process_task, task, output_pt_dir, edge_dump_dir, builder_config): task
                    for task in tasks
                }
                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        model, error, dim = future.result()
                    except Exception as exc:  # pragma: no cover
                        error = str(exc)
                        model = task.model_key
                        dim = None
                    if error:
                        failures.append((model, error, task.log_path))
                        _write_model_log(task, "FAILURE", error)
                    else:
                        success += 1
                        feature_dim = feature_dim or dim
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
    run_lines.append("")
    run_lines.append(f"Total candidates : {len(tasks)}")
    run_lines.append(f"Successes        : {success}")
    run_lines.append(f"Failures         : {len(failures)}")
    run_lines.append(f"Elapsed          : {elapsed:0.2f} s")
    if feature_dim is not None:
        run_lines.append(f"Edge feature dim : {feature_dim}")
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
        edge_feature_dim=feature_dim,
    )
