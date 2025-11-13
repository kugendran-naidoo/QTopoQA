from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ..base import (
    InterfaceFeatureModule,
    build_metadata,
    require_int,
    require_positive_float,
    require_positive_int,
)
from ..registry import register_feature_module
from ...lib import interface_runner


@register_feature_module
class DefaultInterfaceModule(InterfaceFeatureModule):
    module_id = "interface/polar_cutoff/v1"
    module_kind = "interface"
    default_alias = "TopoQA default 10A cut-off"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Polar interface detection using distance cutoff.",
        description=(
            "Identifies interface residues across chains using the polar interface "
            "routine (new_calculate_interface.process_pdb_file) with configurable "
            "distance cutoff and coordinate rounding."
        ),
        inputs=("pdb_file",),
        outputs=("interface_file",),
        parameters={
            "cutoff": "Distance cutoff in Å for interface residue detection.",
            "coordinate_decimals": (
                "Decimal places for coordinate rounding in output files (-1 skips rounding)."
            ),
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "cutoff": 10.0,
            "coordinate_decimals": -1,
            "jobs": 4,
        },
    )

    def extract_interfaces(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        work_dir: Path,
        log_dir: Path,
    ) -> Tuple[int, List[Tuple[Path, Path, str]]]:
        params = self.params
        return interface_runner.run_interface_stage(
            pdb_paths=pdb_paths,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            log_dir=log_dir,
            cutoff=float(params["cutoff"]),
            coordinate_decimals=int(params["coordinate_decimals"]),
            jobs=params.get("jobs"),
        )

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        cutoff = params.get("cutoff")
        if cutoff is not None:
            params["cutoff"] = require_positive_float(cutoff, "interface.params.cutoff")
        coordinate_decimals = params.get("coordinate_decimals")
        if coordinate_decimals is not None:
            decimals = require_int(coordinate_decimals, "interface.params.coordinate_decimals")
            if decimals < -1:
                raise ValueError("interface.params.coordinate_decimals must be >= -1.")
            params["coordinate_decimals"] = decimals
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "interface.params.jobs")

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        template = super().config_template()
        comments = dict(template.get("param_comments", {}))
        comments["coordinate_decimals"] = "skip rounding to keep raw coords"
        template["param_comments"] = comments
        return template
