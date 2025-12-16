from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from ..base import (
    TopologyFeatureModule,
    build_metadata,
    require_bool,
    require_float,
    require_positive_int,
)
from ..registry import register_feature_module
from ...lib.new_topological_features import TopologicalConfig, _feature_column_labels
from ...lib.progress import StageProgress
from ...lib.topology_runner import _load_interface_descriptors, round_topology_frame

# Reuse the graph_builder logger for progress/ETA messages.
LOG = logging.getLogger("graph_builder")

# Default element filters mirror the baseline PH setup to keep the dim at 140.
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


def _process_null_task(
    pdb_path: Path,
    interface_path: Path,
    output_path: Path,
    zero_cols: Sequence[str],
    one_cols: Sequence[str],
    fill_value: float,
    sort_artifacts: bool,
    round_decimals: Optional[int],
) -> Dict[str, Any]:
    descriptors, error = _load_interface_descriptors(interface_path)
    if error:
        return {"error": error, "residue_count": 0}

    ids = [desc.to_string() for desc in descriptors]
    if sort_artifacts:
        ids = sorted(ids)

    numeric_cols = list(zero_cols) + list(one_cols)
    frame = pd.DataFrame([[fill_value] * len(numeric_cols) for _ in ids], columns=numeric_cols)
    frame.insert(0, "ID", ids)
    round_topology_frame(frame, round_decimals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return {"error": None, "residue_count": len(ids)}


@register_feature_module
class PersistenceNullTopologyModule(TopologyFeatureModule):
    """
    Ablation module that emits constant topology features (no PH signal).

    Produces a deterministic CSV with the same schema as the baseline PH block
    (ID + 140 numeric columns by default), filled with a constant value
    (default 0.0). Useful for “no-topology” controls while keeping downstream
    dimensions and schemas unchanged.
    """

    module_id = "topology/persistence_null/v1"
    module_kind = "topology"
    default_alias = "Null topology (constant zeros) = 140D ablation"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Topology ablation: constant features with PH-compatible schema.",
        description=(
            "Writes per-residue constant vectors matching the baseline PH schema "
            "(default 140D) to enable no-topology ablation experiments without "
            "changing downstream dimensions or schemas."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "element_filters": "Element filter labels to determine column layout (matches PH defaults).",
            "constant_value": "Fill value for all topology features (float).",
            "jobs": "Optional override for worker count.",
        },
        defaults={
            "element_filters": _DEFAULT_ELEMENT_FILTERS,
            "constant_value": 0.0,
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
        element_filters: Tuple[Tuple[str, ...], ...] = tuple(tuple(x) for x in params["element_filters"])
        constant_value = float(params.get("constant_value", 0.0))
        jobs = params.get("jobs")

        return _run_null_topology_stage(
            pdb_paths=pdb_paths,
            dataset_dir=dataset_dir,
            interface_dir=interface_dir,
            work_dir=work_dir,
            log_dir=log_dir,
            element_filters=element_filters,
            constant_value=constant_value,
            jobs=jobs,
            sort_artifacts=sort_artifacts,
            round_decimals=round_decimals,
        )

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        element_filters = params.get("element_filters")
        if element_filters is not None:
            if isinstance(element_filters, str):
                raise ValueError("topology.params.element_filters must be a list/tuple, not a single string.")
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

        constant_value = params.get("constant_value")
        if constant_value is not None:
            params["constant_value"] = require_float(constant_value, "topology.params.constant_value")

        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        params = dict(cls._metadata.defaults)
        param_comments = {
            "element_filters": "Element label layout (matches baseline PH defaults); affects column names only.",
            "constant_value": "Fill value for all topology features (default 0.0).",
            "jobs": "Optional worker override (simple IO-bound stage; sequential is usually fine).",
        }
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "notes": {"feature_dim": FEATURE_DIM, "ablation": "No topology signal; constant output."},
        }


def _run_null_topology_stage(
    pdb_paths: Iterable[Path],
    dataset_dir: Path,
    interface_dir: Path,
    work_dir: Path,
    log_dir: Path,
    element_filters: Sequence[Sequence[str]],
    constant_value: float,
    jobs: Optional[int],
    sort_artifacts: bool = True,
    round_decimals: Optional[int] = None,
) -> Dict[str, object]:
    topology_dir = work_dir / "topology"
    topology_dir.mkdir(parents=True, exist_ok=True)
    topology_log_dir = log_dir / "topology_logs"
    topology_log_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[Path, Path, Path, Path]] = []
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
        interface_path = interface_dir / relative_path.parent / f"{pdb_path.stem}.interface.txt"
        topology_output_path = topology_output_parent / f"{pdb_path.stem}.topology.csv"
        topology_log_path = topology_log_parent / f"{pdb_path.stem}.log"
        tasks.append((pdb_path, interface_path, topology_output_path, topology_log_path))

    worker_count = max(1, int(jobs)) if jobs else 1
    zero_cols, one_cols = _feature_column_labels(TopologicalConfig(element_filters=element_filters))

    success = 0
    failures: List[Tuple[Path, Path, str]] = []
    elapsed = 0.0
    total = len(tasks)
    dataset_label = dataset_dir.name
    progress = StageProgress("Topology", total, dataset_name=dataset_label)

    if tasks:
        start = time.perf_counter()

        def _log_progress():
            if total <= 0:
                return
            elapsed_local = time.perf_counter() - start
            pct = (success / total) * 100 if total else 100.0
            rate = success / elapsed_local if elapsed_local > 0 else 0.0
            remaining = (total - success) / rate if rate > 0 else 0.0
            LOG.info(
                "[Topology - %s] %.1f%% complete (%d/%d) ETA %.1f min",
                dataset_label,
                pct,
                success,
                total,
                remaining / 60.0 if remaining else 0.0,
            )

        if worker_count <= 1:
            for pdb_path, interface_path, output_path, log_path in tasks:
                result = _process_null_task(
                    pdb_path=pdb_path,
                    interface_path=interface_path,
                    output_path=output_path,
                    zero_cols=zero_cols,
                    one_cols=one_cols,
                    fill_value=constant_value,
                    sort_artifacts=sort_artifacts,
                    round_decimals=round_decimals,
                )
                if result["error"]:
                    failures.append((pdb_path, log_path, result["error"]))
                    log_path.write_text(
                        f"PDB: {pdb_path}\nStatus: FAILURE\nError: {result['error']}\n", encoding="utf-8"
                    )
                else:
                    success += 1
                    log_path.write_text(
                        "\n".join(
                            [
                                f"PDB: {pdb_path}",
                                "Status: SUCCESS",
                                f"Residues processed: {result['residue_count']}",
                                f"Output file: {output_path}",
                                "Note: null topology (constant features).",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    if success % 5 == 0:
                        _log_progress()
                progress.increment()
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_meta = {
                    executor.submit(
                        _process_null_task,
                        pdb_path,
                        interface_path,
                        output_path,
                        zero_cols,
                        one_cols,
                        constant_value,
                        sort_artifacts,
                        round_decimals,
                    ): (pdb_path, interface_path, output_path, log_path)
                    for pdb_path, interface_path, output_path, log_path in tasks
                }
                for future in as_completed(future_to_meta):
                    pdb_path, _, output_path, log_path = future_to_meta[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover
                        failures.append((pdb_path, log_path, str(exc)))
                        log_path.write_text(f"PDB: {pdb_path}\nStatus: FAILURE\nError: {exc}\n", encoding="utf-8")
                    else:
                        if result["error"]:
                            failures.append((pdb_path, log_path, result["error"]))
                            log_path.write_text(
                                f"PDB: {pdb_path}\nStatus: FAILURE\nError: {result['error']}\n", encoding="utf-8"
                            )
                        else:
                            success += 1
                            log_path.write_text(
                                "\n".join(
                                    [
                                        f"PDB: {pdb_path}",
                                        "Status: SUCCESS",
                                        f"Residues processed: {result['residue_count']}",
                                        f"Output file: {output_path}",
                                        "Note: null topology (constant features).",
                                    ]
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                            if success % 5 == 0:
                                _log_progress()
                    finally:
                        progress.increment()
        elapsed = time.perf_counter() - start

    return {
        "output_dir": topology_dir,
        "log_dir": topology_log_dir,
        "success": success,
        "failures": failures,
        "elapsed": elapsed,
        "processed": len(tasks),
    }
