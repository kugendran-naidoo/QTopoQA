from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..base import (
    TopologyFeatureModule,
    build_metadata,
    require_bool,
    require_positive_float,
    require_positive_int,
)
from ..registry import register_feature_module
from ...lib.new_topological_features import TopologicalConfig, _feature_column_labels
from ...lib.progress import StageProgress
from ...lib import topology_runner

LOG = logging.getLogger("graph_builder")

_DEFAULT_ELEMENT_FILTERS: Tuple[Tuple[str, ...], ...] = (
    ("C",),
    ("N",),
    ("O",),
    ("C", "N"),
    ("C", "O"),
    ("N", "O"),
    ("C", "N", "O"),
)

_ZERO_COLS, _ONE_COLS = _feature_column_labels(TopologicalConfig(element_filters=_DEFAULT_ELEMENT_FILTERS))
FEATURE_DIM = len(_ZERO_COLS) + len(_ONE_COLS)

_DEFAULT_SEED = 1337
_SHUFFLE_SCOPES = {"per_protein", "global"}
_SHUFFLE_MODES = {"per_row", "per_feature"}


def _stable_seed(base_seed: int, token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:16], 16)) % (2**32)


def _hash_permutation(items: np.ndarray, *, tag: Optional[str] = None) -> str:
    hasher = hashlib.sha256()
    if tag:
        hasher.update(tag.encode("utf-8"))
    hasher.update(items.tobytes())
    return hasher.hexdigest()


def _permute_frame(
    frame: pd.DataFrame,
    numeric_cols: Sequence[str],
    shuffle_mode: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, str]:
    if shuffle_mode == "per_row":
        perm = rng.permutation(len(frame))
        permuted = frame.copy()
        permuted.loc[:, numeric_cols] = frame.loc[:, numeric_cols].to_numpy()[perm, :]
        perm_hash = _hash_permutation(perm)
        return permuted, perm_hash

    if shuffle_mode == "per_feature":
        permuted = frame.copy()
        hasher = hashlib.sha256()
        for col in numeric_cols:
            perm = rng.permutation(len(frame))
            hasher.update(col.encode("utf-8"))
            hasher.update(perm.tobytes())
            permuted[col] = frame[col].to_numpy()[perm]
        return permuted, hasher.hexdigest()

    raise ValueError(f"Unsupported shuffle_mode: {shuffle_mode}")


def _append_permutation_log(
    log_path: Path,
    *,
    shuffle_scope: str,
    shuffle_mode: str,
    seed: int,
    rows: int,
    cols: int,
    perm_hash: str,
    seed_override: bool,
) -> None:
    lines = [
        "Permutation null details:",
        f"  scope={shuffle_scope}",
        f"  mode={shuffle_mode}",
        f"  seed={seed}",
        f"  seed_override={seed_override}",
        f"  rows={rows}",
        f"  cols={cols}",
        f"  perm_hash={perm_hash}",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines) + "\n")


def _process_permutation_task(
    topology_path: Path,
    log_path: Path,
    shuffle_mode: str,
    seed: int,
    shuffle_scope: str,
    seed_override: bool,
) -> Dict[str, Any]:
    try:
        frame = pd.read_csv(topology_path)
    except Exception as exc:
        return {"error": f"Failed to read topology CSV {topology_path}: {exc}"}

    if "ID" not in frame.columns:
        return {"error": f"Topology CSV missing ID column: {topology_path}"}

    numeric_cols = [col for col in frame.columns if col != "ID"]
    if not numeric_cols:
        return {"error": f"Topology CSV has no numeric columns: {topology_path}"}

    rng = np.random.default_rng(seed)
    permuted, perm_hash = _permute_frame(frame, numeric_cols, shuffle_mode, rng)
    permuted.to_csv(topology_path, index=False)
    _append_permutation_log(
        log_path,
        shuffle_scope=shuffle_scope,
        shuffle_mode=shuffle_mode,
        seed=seed,
        rows=len(permuted),
        cols=len(numeric_cols),
        perm_hash=perm_hash,
        seed_override=seed_override,
    )
    return {"error": None}


def _build_topology_tasks(
    pdb_paths: Iterable[Path],
    dataset_dir: Path,
    work_dir: Path,
    log_dir: Path,
) -> List[Tuple[Path, Path]]:
    tasks: List[Tuple[Path, Path]] = []
    topology_dir = work_dir / "topology"
    topology_log_dir = log_dir / "topology_logs"
    for pdb_path in sorted(pdb_paths):
        try:
            relative = pdb_path.relative_to(dataset_dir)
            relative_path = Path(relative)
        except ValueError:
            relative_path = Path(pdb_path.name)
        topology_output_parent = topology_dir / relative_path.parent
        topology_log_parent = topology_log_dir / relative_path.parent
        topology_output_parent.mkdir(parents=True, exist_ok=True)
        topology_log_parent.mkdir(parents=True, exist_ok=True)
        topology_output_path = topology_output_parent / f"{pdb_path.stem}.topology.csv"
        topology_log_path = topology_log_parent / f"{pdb_path.stem}.log"
        tasks.append((topology_output_path, topology_log_path))
    return tasks


def _permute_per_protein(
    tasks: List[Tuple[Path, Path]],
    *,
    shuffle_mode: str,
    base_seed: int,
    seed_override: bool,
    jobs: Optional[int],
    dataset_label: str,
) -> List[Tuple[Path, Path, str]]:
    failures: List[Tuple[Path, Path, str]] = []
    worker_count = max(1, int(jobs)) if jobs else 1
    progress = StageProgress("TopologyPermute", len(tasks), dataset_name=dataset_label)

    def _seed_for_path(path: Path) -> int:
        return _stable_seed(base_seed, str(path))

    if worker_count <= 1:
        for topology_path, log_path in tasks:
            seed = _seed_for_path(topology_path)
            result = _process_permutation_task(
                topology_path,
                log_path,
                shuffle_mode,
                seed,
                "per_protein",
                seed_override,
            )
            if result["error"]:
                failures.append((topology_path, log_path, result["error"]))
            progress.increment()
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_meta = {
                executor.submit(
                    _process_permutation_task,
                    topology_path,
                    log_path,
                    shuffle_mode,
                    _seed_for_path(topology_path),
                    "per_protein",
                    seed_override,
                ): (topology_path, log_path)
                for topology_path, log_path in tasks
            }
            for future in as_completed(future_to_meta):
                topology_path, log_path = future_to_meta[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover
                    failures.append((topology_path, log_path, str(exc)))
                else:
                    if result["error"]:
                        failures.append((topology_path, log_path, result["error"]))
                finally:
                    progress.increment()
    return failures


def _permute_global(
    tasks: List[Tuple[Path, Path]],
    *,
    shuffle_mode: str,
    seed: int,
    seed_override: bool,
) -> List[Tuple[Path, Path, str]]:
    frames: List[pd.DataFrame] = []
    numeric_cols: Optional[List[str]] = None
    for topology_path, log_path in tasks:
        try:
            frame = pd.read_csv(topology_path)
        except Exception as exc:
            return [(topology_path, log_path, f"Failed to read topology CSV {topology_path}: {exc}")]
        if "ID" not in frame.columns:
            return [(topology_path, log_path, f"Topology CSV missing ID column: {topology_path}")]
        cols = [col for col in frame.columns if col != "ID"]
        if not cols:
            return [(topology_path, log_path, f"Topology CSV has no numeric columns: {topology_path}")]
        if numeric_cols is None:
            numeric_cols = cols
        elif cols != numeric_cols:
            return [(topology_path, log_path, "Topology CSV columns inconsistent across files for global shuffle.")]
        frames.append(frame)

    if numeric_cols is None:
        return []

    rng = np.random.default_rng(seed)
    lengths = [len(frame) for frame in frames]
    if shuffle_mode == "per_row":
        combined = np.concatenate([frame.loc[:, numeric_cols].to_numpy() for frame in frames], axis=0)
        perm = rng.permutation(len(combined))
        combined = combined[perm, :]
        perm_hash = _hash_permutation(perm)
        offset = 0
        for (topology_path, log_path), frame, count in zip(tasks, frames, lengths):
            chunk = combined[offset : offset + count]
            frame.loc[:, numeric_cols] = chunk
            frame.to_csv(topology_path, index=False)
            _append_permutation_log(
                log_path,
                shuffle_scope="global",
                shuffle_mode=shuffle_mode,
                seed=seed,
                rows=count,
                cols=len(numeric_cols),
                perm_hash=perm_hash,
                seed_override=seed_override,
            )
            offset += count
    else:
        combined = np.concatenate([frame.loc[:, numeric_cols].to_numpy() for frame in frames], axis=0)
        hasher = hashlib.sha256()
        for col_idx, col_name in enumerate(numeric_cols):
            perm = rng.permutation(len(combined))
            hasher.update(col_name.encode("utf-8"))
            hasher.update(perm.tobytes())
            combined[:, col_idx] = combined[perm, col_idx]
        perm_hash = hasher.hexdigest()
        offset = 0
        for (topology_path, log_path), frame, count in zip(tasks, frames, lengths):
            chunk = combined[offset : offset + count]
            frame.loc[:, numeric_cols] = chunk
            frame.to_csv(topology_path, index=False)
            _append_permutation_log(
                log_path,
                shuffle_scope="global",
                shuffle_mode=shuffle_mode,
                seed=seed,
                rows=count,
                cols=len(numeric_cols),
                perm_hash=perm_hash,
                seed_override=seed_override,
            )
            offset += count

    return []


@register_feature_module
class PersistencePermutationNullTopologyModule(TopologyFeatureModule):
    """
    Ablation module that permutes PH topology features to destroy residue mapping.

    The topology is computed identically to persistence_basic/v1, then features
    are permuted (per-row by default) so marginal distributions are preserved
    but per-residue topology signal is removed.
    """

    module_id = "topology/persistence_permutation_null/v2"
    module_kind = "topology"
    default_alias = "Permutation null topology (per-row, per-protein) = 140D ablation"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Topology ablation: permuted PH features with preserved marginal distributions.",
        description=(
            "Computes the baseline PH features and then permutes them to break "
            "the residue-to-topology association while preserving feature "
            "distributions. Intended as a stronger, more defensible ablation."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "neighbor_distance": "Neighbourhood radius in Å for complex filtration.",
            "filtration_cutoff": "Maximum filtration value in Å.",
            "min_persistence": "Minimum persistence threshold for features.",
            "dedup_sort": "Enable deduplication & sorting of persistence pairs.",
            "element_filters": "Sequence of element subsets considered for statistics.",
            "shuffle_scope": "Permutation scope: per_protein (default) or global.",
            "shuffle_mode": "Permutation mode: per_row (default) or per_feature.",
            "seed": "Base seed for deterministic shuffling.",
            "allow_seed_override": "Allow overriding the default seed.",
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "neighbor_distance": 8.0,
            "filtration_cutoff": 8.0,
            "min_persistence": 0.01,
            "dedup_sort": False,
            "element_filters": _DEFAULT_ELEMENT_FILTERS,
            "shuffle_scope": "per_protein",
            "shuffle_mode": "per_row",
            "seed": _DEFAULT_SEED,
            "allow_seed_override": False,
            "jobs": 16,
        },
    )

    def generate_topology(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        interface_dir: Path,
        work_dir: Path,
        log_dir: Path,
        sort_artifacts: bool = True,
        round_decimals: Optional[int] = None,
    ):
        params = self.params
        element_filters: Sequence[Sequence[str]] = tuple(tuple(x) for x in params["element_filters"])
        shuffle_scope = str(params.get("shuffle_scope", "per_protein"))
        shuffle_mode = str(params.get("shuffle_mode", "per_row"))
        allow_seed_override = bool(params.get("allow_seed_override", False))
        base_seed = int(params.get("seed", _DEFAULT_SEED))
        if not allow_seed_override:
            base_seed = _DEFAULT_SEED
        jobs = params.get("jobs")

        topology_result = topology_runner.run_topology_stage(
            pdb_paths=pdb_paths,
            dataset_dir=dataset_dir,
            interface_dir=interface_dir,
            work_dir=work_dir,
            log_dir=log_dir,
            neighbor_distance=float(params["neighbor_distance"]),
            filtration_cutoff=float(params["filtration_cutoff"]),
            min_persistence=float(params["min_persistence"]),
            element_filters=element_filters,
            dedup_sort=bool(params["dedup_sort"]),
            jobs=jobs,
            sort_artifacts=sort_artifacts,
            round_decimals=round_decimals,
        )

        tasks = _build_topology_tasks(pdb_paths, dataset_dir, work_dir, log_dir)
        dataset_label = dataset_dir.name
        start = time.perf_counter()

        if shuffle_scope == "per_protein":
            failures = _permute_per_protein(
                tasks,
                shuffle_mode=shuffle_mode,
                base_seed=base_seed,
                seed_override=allow_seed_override,
                jobs=jobs,
                dataset_label=dataset_label,
            )
        else:
            failures = _permute_global(
                tasks,
                shuffle_mode=shuffle_mode,
                seed=base_seed,
                seed_override=allow_seed_override,
            )

        elapsed = time.perf_counter() - start
        if failures:
            topology_result["failures"] = topology_result.get("failures", []) + failures
        topology_result["elapsed"] = topology_result.get("elapsed", 0.0) + elapsed
        return topology_result

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        for key in ("neighbor_distance", "filtration_cutoff"):
            value = params.get(key)
            if value is not None:
                params[key] = require_positive_float(value, f"topology.params.{key}")
        min_persistence = params.get("min_persistence")
        if min_persistence is not None:
            params["min_persistence"] = require_positive_float(
                min_persistence, "topology.params.min_persistence", allow_zero=True
            )
        dedup_sort = params.get("dedup_sort")
        if dedup_sort is not None:
            params["dedup_sort"] = require_bool(dedup_sort, "topology.params.dedup_sort")

        element_filters = params.get("element_filters")
        if element_filters is not None:
            if isinstance(element_filters, str):
                raise ValueError(
                    "topology.params.element_filters must be a YAML list, not a single string literal."
                )
            if not isinstance(element_filters, (list, tuple)):
                raise ValueError("topology.params.element_filters must be a list/tuple of sequences.")
            normalised = []
            for entry in element_filters:
                if not isinstance(entry, (list, tuple)):
                    raise ValueError("Each element_filters entry must be a list/tuple of element symbols.")
                if not entry:
                    raise ValueError("Element filter sequences must contain at least one symbol.")
                symbols = []
                for symbol in entry:
                    if not isinstance(symbol, str) or not symbol.strip():
                        raise ValueError("Element filter symbols must be non-empty strings.")
                    symbols.append(symbol.strip())
                normalised.append(tuple(symbols))
            params["element_filters"] = tuple(normalised)

        shuffle_scope = params.get("shuffle_scope", "per_protein")
        if shuffle_scope not in _SHUFFLE_SCOPES:
            raise ValueError(f"topology.params.shuffle_scope must be one of {sorted(_SHUFFLE_SCOPES)}")

        shuffle_mode = params.get("shuffle_mode", "per_row")
        if shuffle_mode not in _SHUFFLE_MODES:
            raise ValueError(f"topology.params.shuffle_mode must be one of {sorted(_SHUFFLE_MODES)}")

        seed = params.get("seed")
        if seed is not None:
            params["seed"] = require_positive_int(seed, "topology.params.seed")

        allow_seed_override = params.get("allow_seed_override")
        if allow_seed_override is not None:
            params["allow_seed_override"] = require_bool(
                allow_seed_override, "topology.params.allow_seed_override"
            )

        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        params = dict(cls._metadata.defaults)
        param_comments = {
            "neighbor_distance": "Neighbourhood radius (Å) for PH; default 8.0",
            "filtration_cutoff": "Max filtration value (Å); default 8.0",
            "min_persistence": "Minimum persistence to keep (default 0.01)",
            "dedup_sort": "Deduplicate/sort persistence pairs for determinism",
            "element_filters": "Element subsets to match PH schema (default TopoQA)",
            "shuffle_scope": "Permutation scope: per_protein (default) or global",
            "shuffle_mode": "Permutation mode: per_row (default) or per_feature",
            "seed": "Base seed for deterministic shuffling (default 1337)",
            "allow_seed_override": "Allow overriding the default seed (default false)",
            "jobs": "Optional worker override for topology computation",
        }
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "notes": {
                "feature_dim": FEATURE_DIM,
                "ablation": "PH features permuted to remove residue mapping.",
                "default_policy": "per_protein/per_row/seed=1337",
            },
        }
