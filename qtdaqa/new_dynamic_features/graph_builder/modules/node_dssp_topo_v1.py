from __future__ import annotations

from pathlib import Path
from typing import Dict

from .base import NodeFeatureModule, build_metadata
from .registry import register_feature_module

try:
    from ..lib import node_runner
except ImportError:  # pragma: no cover
    from lib import node_runner  # type: ignore


@register_feature_module
class DSSPTopologyNodeModule(NodeFeatureModule):
    module_id = "node/dssp_topo_merge/v1"
    module_kind = "node"
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
        defaults={"drop_na": False, "jobs": None},
    )

    def generate_nodes(
        self,
        dataset_dir: Path,
        structure_map: Dict[str, Path],
        interface_dir: Path,
        topology_dir: Path,
        work_dir: Path,
        log_dir: Path,
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
        )
