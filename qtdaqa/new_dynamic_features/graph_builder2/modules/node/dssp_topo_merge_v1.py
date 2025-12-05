from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..base import NodeFeatureModule, build_metadata, require_bool, require_positive_int
from ..registry import register_feature_module
from ...lib import node_runner

NODE_BASE_DIM = 32  # DSSP/basic features
TOPO_DIM_HINT = 140  # default PH topo_dim
FEATURE_DIM_HINT = NODE_BASE_DIM + TOPO_DIM_HINT

@register_feature_module
class DSSPTopologyNodeModule(NodeFeatureModule):
    module_id = "node/dssp_topo_merge/v1"
    module_kind = "node"
    default_alias = "140D PH (topology summaries on interface coords) + 32D DSSP = Node 172D | TopoQA"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="DSSP-derived node features merged with topology statistics.",
        description=(
            "Runs the DSSP-based node feature extractor (node_fea) and merges "
            "results with persistent homology statistics aligned by residue IDs."
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
        params = self.params
        return node_runner.run_node_stage(
            dataset_dir=dataset_dir,
            structure_map=structure_map,
            interface_dir=interface_dir,
            topology_dir=topology_dir,
            work_dir=work_dir,
            log_dir=log_dir,
            drop_na=bool(params["drop_na"]),
            jobs=params.get("jobs"),
            sort_artifacts=sort_artifacts,
        )

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        drop_na = params.get("drop_na")
        if drop_na is not None:
            params["drop_na"] = require_bool(drop_na, "node.params.drop_na")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "node.params.jobs")

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        comments = {
            "drop_na": "Drop rows with NA after merge (default on); matches fea_df_clean behavior in inference.",
            "jobs": "Optional worker override (CLI --jobs takes precedence over config/module default).",
        }
        template["param_comments"] = comments
        template.setdefault("notes", {})
        template["notes"].update(
            {
                "feature_dim_hint": f"{NODE_BASE_DIM} DSSP + topo_dim (default topo_dim ~{TOPO_DIM_HINT} → node_dim ~{FEATURE_DIM_HINT})",
                "determinism": "Merge preserves deterministic ordering when sort_artifacts is enabled.",
                "jobs_precedence": "CLI --jobs > config default_jobs > module default.",
            }
        )
        return template
