from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple, Optional

from ..base import (
    TopologyFeatureModule,
    build_metadata,
    require_bool,
    require_float,
    require_positive_float,
    require_positive_int,
)
from ..registry import register_feature_module
from ...lib import topology_runner

FEATURE_DIM = 140  # persistent homology block (f0/f1 stats across element filters)


@register_feature_module
class PersistenceTopologyModule(TopologyFeatureModule):
    module_id = "topology/persistence_basic/v1"
    module_kind = "topology"
    default_alias = "Betti 0 and Betti 1 topology summaries on interface coords = 140D PH | TopoQA"
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
            sort_artifacts=sort_artifacts,
            round_decimals=round_decimals,
        )

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
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "topology.params.jobs")

    @classmethod
    def config_template(cls) -> Dict[str, object]:
        params = dict(cls._metadata.defaults)
        param_comments = {
            "neighbor_distance": "Neighbourhood radius (Å) for building complexes; default 8.0",
            "filtration_cutoff": "Max filtration value (Å); cost grows with larger cutoff (default 8.0)",
            "min_persistence": "Minimum persistence to keep (default 0.01)",
            "dedup_sort": "Deduplicate/sort persistence pairs for determinism; may add slight cost",
            "element_filters": "List of element subsets (e.g., C/N/O combos); more filters = higher cost",
            "jobs": "Optional override for worker count",
        }
        return {
            "module": cls.module_id,
            "alias": cls.default_alias,
            "summary": cls._metadata.summary,
            "description": cls._metadata.description,
            "params": params,
            "param_comments": param_comments,
            "notes": {"feature_dim": FEATURE_DIM},
        }
