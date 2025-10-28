from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from .base import InterfaceFeatureModule, build_metadata
from .registry import register_feature_module

try:
    from ..lib import interface_runner
except ImportError:  # pragma: no cover
    from lib import interface_runner  # type: ignore


@register_feature_module
class DefaultInterfaceModule(InterfaceFeatureModule):
    module_id = "interface/polar_cutoff/v1"
    module_kind = "interface"
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
            "coordinate_decimals": "Decimal places for coordinate rounding in output files.",
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "cutoff": 14.0,
            "coordinate_decimals": 3,
            "jobs": None,
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
