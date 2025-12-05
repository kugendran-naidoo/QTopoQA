from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..base import NodeFeatureModule, build_metadata, require_bool, require_positive_int
from ..registry import register_feature_module
from ...lib.stage_common import gather_files, normalise_interface_name, normalise_topology_name, relative_key
from ...lib.progress import StageProgress
from ...lib.node_utils import canonical_id_order
from ...lib import node_features

NODE_BASE_DIM = 32  # DSSP/basic block
TOPO_DIM_HINT = 140  # default PH topo dim (dynamic in practice)

@register_feature_module
class DSSPTopologyMergePassthrough(NodeFeatureModule):
    """
    Merge DSSP/basic features with all topology columns (passthrough).

    Unlike node/dssp_topo_merge/v1 (which assumes the legacy 140D topology),
    this module consumes whatever topology columns are present (non-ID),
    scales them per-file, and appends the DSSP/basic 32D block. Intended to
    preserve hybrid topology vectors (e.g., PH+Lap 172D) without dropping
    columns. Output columns: ID, DSSP/basic 32D, then all topology columns
    in file order.
    """

    module_id = "node/dssp_topo_merge_passthrough/v1"
    module_kind = "node"
    default_alias = "Passthrough (all dims from topology module) + 32D DSSP = Node Dynamic Dims"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="DSSP/basic features merged with all topology columns (passthrough).",
        description=(
            "Runs DSSP/basic feature extraction and merges with the provided topology CSV "
            "using all non-ID topology columns (no hardcoded list). Topology columns are "
            "MinMax scaled per file; output columns ordered as ID, DSSP/basic 32D, then "
            "topology columns in file order. Suitable for hybrid topology (PH+Lap) or any "
            "dynamic topo layout."
        ),
        inputs=("pdb_file", "interface_file", "topology_csv"),
        outputs=("node_csv",),
        parameters={
            "drop_na": "Drop rows containing NA values after merging.",
            "jobs": "Optional override for thread worker count.",
        },
        defaults={"drop_na": True, "jobs": 16},
    )

    def generate_nodes(
        self,
        dataset_dir: Path,
        structure_map: Dict[str, Path],
        interface_dir: Path,
        topology_dir: Path,
        work_dir: Path,
        log_dir: Path,
        sort_artifacts: bool = True,
    ):
        node_dir = work_dir / "node_features"
        node_dir.mkdir(parents=True, exist_ok=True)
        node_log_dir = log_dir / "node_feature_logs"
        node_log_dir.mkdir(parents=True, exist_ok=True)

        interface_map = gather_files(interface_dir, ("*.interface.txt", "*.txt"), normalise_interface_name)
        topology_map = gather_files(topology_dir, ("*.topology.csv", "*.csv"), normalise_topology_name)
        node_tasks: List[Tuple[str, Path, Path, Path, Path]] = []
        shared = sorted(set(structure_map) & set(interface_map) & set(topology_map))
        for key in shared:
            model_name = Path(key).name
            output_rel = Path(relative_key(interface_dir, interface_map[key], model_name)).with_suffix(".csv")
            output_path = (node_dir / output_rel).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            log_rel = output_rel.with_suffix(".log")
            log_path = (node_log_dir / log_rel).resolve()
            node_tasks.append(
                (
                    key,
                    structure_map[key],
                    interface_map[key],
                    topology_map[key],
                    output_path,
                    log_path,
                )
            )

        success = 0
        failures: List[Tuple[str, Path, str]] = []
        elapsed = 0.0

        progress = StageProgress("Node (passthrough)", len(node_tasks), dataset_name=dataset_dir.name)

        if node_tasks:
            import time
            start = time.perf_counter()
            worker_count = max(1, int(self.params.get("jobs") or 1))
            worker_count = min(worker_count, len(node_tasks))
            if worker_count <= 1:
                for task in node_tasks:
                    ok, failure = self._process_task(task, sort_artifacts)
                    if ok:
                        success += 1
                    elif failure:
                        failures.append(failure)
                    progress.increment()
            else:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {executor.submit(self._process_task, task, sort_artifacts): task for task in node_tasks}
                    for future in as_completed(future_map):
                        ok, failure = future.result()
                        if ok:
                            success += 1
                        elif failure:
                            failures.append(failure)
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

    def _process_task(
        self,
        task: Tuple[str, Path, Path, Path, Path, Path],
        sort_artifacts: bool,
    ) -> Tuple[bool, Tuple[str, Path, str] | None]:
        model_key, pdb_path, interface_path, topo_path, output_path, log_path = task
        log_lines: List[str] = []
        try:
            dssp_df = self._compute_dssp(model_key, pdb_path, interface_path)
            topo_df = pd.read_csv(topo_path)
            if "ID" not in topo_df.columns:
                raise ValueError(f"Topology file {topo_path} missing ID column.")
            topo_cols = [col for col in topo_df.columns if col != "ID"]
            if topo_cols:
                scaler = MinMaxScaler()
                topo_df[topo_cols] = scaler.fit_transform(topo_df[topo_cols])
            merged = pd.merge(dssp_df, topo_df, on="ID", how="inner")
            if self.params.get("drop_na", True):
                merged = merged.replace("NA", pd.NA).dropna()
            if merged.empty:
                raise ValueError("Merged node feature frame is empty.")
            if sort_artifacts:
                merged = merged.sort_values(by=["ID"], kind="mergesort").reset_index(drop=True)
            merged.to_csv(output_path, index=False)
            log_lines.extend(
                [
                    f"Model key: {model_key}",
                    f"PDB: {pdb_path}",
                    f"Interface: {interface_path}",
                    f"Topology: {topo_path}",
                    f"Output CSV: {output_path}",
                    f"DSSP/basic dims: {self._dssp_dim}",
                    f"Topology dims: {len(topo_cols)}",
                    f"Node feature dim: {self._dssp_dim + len(topo_cols)}",
                    "Status: SUCCESS",
                ]
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return True, None
        except Exception as exc:  # pragma: no cover
            log_lines.extend(
                [
                    f"Model key: {model_key}",
                    f"PDB: {pdb_path}",
                    f"Topology: {topo_path}",
                    "Status: FAILURE",
                    f"Error: {exc}",
                ]
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            return False, (model_key, log_path, str(exc))

    def _compute_dssp(self, model_key: str, pdb_path: Path, interface_path: Path) -> pd.DataFrame:
        model_name = Path(model_key).name
        extractor = node_features.node_fea(model_name, str(pdb_path.parent), str(interface_path.parent), str(interface_path.parent))
        dssp_df = extractor.run_dssp(str(pdb_path))
        self._dssp_dim = len([col for col in dssp_df.columns if col != "ID"])
        return dssp_df

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        drop_na = params.get("drop_na")
        if drop_na is not None:
            params["drop_na"] = require_bool(drop_na, "node.params.drop_na")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "node.params.jobs")

    @classmethod
    def list_params(cls) -> Dict[str, str]:
        return dict(cls._metadata.parameters)

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        template["alias"] = cls.default_alias
        template["param_comments"] = {
            "drop_na": "Drop rows with NA after merge (default on); aligns with inference cleaning.",
            "jobs": "Optional worker override (CLI --jobs takes precedence).",
        }
        template.setdefault("notes", {})
        template["notes"].update(
            {
                "feature_dim_hint": f"dynamic: {NODE_BASE_DIM} DSSP + topo_dim (e.g., ~{NODE_BASE_DIM + TOPO_DIM_HINT} with 140D PH)",
                "determinism": "Merge preserves deterministic ordering when sort_artifacts is enabled.",
                "jobs_precedence": "CLI --jobs > config default_jobs > module default.",
            }
        )
        return template
