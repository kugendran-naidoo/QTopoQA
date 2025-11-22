from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

from .base import TopologyFeatureModule, build_metadata
from .registry import register_feature_module

try:
    from ..lib import topology_runner
except ImportError:  # pragma: no cover
    from lib import topology_runner  # type: ignore


@register_feature_module
class PersistenceTopologyModule(TopologyFeatureModule):
    module_id = "topology/persistence_basic/v1"
    module_kind = "topology"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="Persistent homology features with configurable distance thresholds.",
        description=(
            "Generates per-residue topological descriptors using the persistent "
            "homology pipeline from new_topological_features. Interface residues "
            "are required as input."
        ),
        inputs=("pdb_file", "interface_file"),
        outputs=("topology_csv",),
        parameters={
            "neighbor_distance": "Neighbourhood radius in Å for complex filtration.",
            "filtration_cutoff": "Maximum filtration value in Å.",
            "min_persistence": "Minimum persistence threshold for features.",
            "dedup_sort": "Enable deduplication & sorting of persistence pairs.",
            "element_filters": "Sequence of element subsets considered for statistics.",
            "jobs": "Optional override for parallel worker count.",
        },
        defaults={
            "neighbor_distance": 8.0,
            "filtration_cutoff": 8.0,
            "min_persistence": 0.01,
            "dedup_sort": False,
            "element_filters": (
                ("C",),
                ("N",),
                ("O",),
                ("C", "N"),
                ("C", "O"),
                ("N", "O"),
                ("C", "N", "O"),
            ),
            "jobs": None,
        },
    )

    def generate_topology(
        self,
        pdb_paths: Iterable[Path],
        dataset_dir: Path,
        interface_dir: Path,
        work_dir: Path,
        log_dir: Path,
    ):
        params = self.params
        element_filters: Sequence[Sequence[str]] = tuple(tuple(x) for x in params["element_filters"])
        return topology_runner.run_topology_stage(
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
            jobs=params.get("jobs"),
        )
