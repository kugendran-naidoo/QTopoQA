"""
Configuration dataclasses and loader for feature toggles and parameters.

All configs are JSON files to allow central control without code edits.
Note: Exact-compatibility and verification are now controlled via CLI flags in
graph_builder.py (not JSON). OtherConfig contains only general knobs (e.g.,
overwrite_graphs, jobs).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cache
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NodeConfig:
    use_rasa: bool = True
    use_phi_psi: bool = True
    use_ss8: bool = True
    use_aa_onehot: bool = True
    use_topological: bool = True
    # Normalize selected features
    normalize_topo: bool = True


@dataclass(frozen=True)
class EdgeConfig:
    cutoff_min: float = 0.0
    cutoff_max: float = 10.0
    include_distance: bool = True
    include_histogram: bool = True
    histogram_bins: int = 10
    # New: choose edge feature mode
    #  - "simple": distance + radial bin (fast)
    #  - "all_atom": legacy-compatible all-atom histogram per edge
    mode: str = "simple"
    # New: control cutoff inclusivity (defaults match legacy strict bounds)
    cutoff_inclusive_min: bool = False
    cutoff_inclusive_max: bool = False
    # New: scaling strategy for edge attributes
    #  - "minmax_sklearn" to mirror legacy; "none" to disable
    scale: str = "minmax_sklearn"


@dataclass(frozen=True)
class TopoConfig:
    neighbor_distance: float = 8.0
    # element sets: list of lists or strings (e.g., ["C"], ["C","N"], ...)
    element_sets: List[List[str]] = None  # type: ignore

    def __post_init__(self):
        # default value via post init for mutables
        if self.element_sets is None:
            object.__setattr__(self, "element_sets", [["C"], ["N"], ["O"], ["C","N"], ["C","O"], ["N","O"], ["C","N","O"]])


@dataclass(frozen=True)
class OtherConfig:
    overwrite_graphs: bool = True
    jobs: int = 4


@cache
def load_json(path: str) -> Dict:
    """
    Load JSON with support for optional // and /* */ comments.
    This allows documenting options directly inside the config files.
    """
    import re
    with open(path, "r") as f:
        text = f.read()
    # strip /* ... */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # strip // ... end-of-line
    text = re.sub(r"(^|\s)//.*$", "", text, flags=re.MULTILINE)
    return json.loads(text)


def _as_bool(d: Dict, key: str, default: bool) -> bool:
    v = d.get(key, default)
    return bool(v)


def load_node_config(path: str) -> NodeConfig:
    d = load_json(path)
    return NodeConfig(
        use_rasa=_as_bool(d, "use_rasa", True),
        use_phi_psi=_as_bool(d, "use_phi_psi", True),
        use_ss8=_as_bool(d, "use_ss8", True),
        use_aa_onehot=_as_bool(d, "use_aa_onehot", True),
        use_topological=_as_bool(d, "use_topological", True),
        normalize_topo=_as_bool(d, "normalize_topo", True),
    )


def load_edge_config(path: str) -> EdgeConfig:
    d = load_json(path)
    return EdgeConfig(
        cutoff_min=float(d.get("cutoff_min", 0.0)),
        cutoff_max=float(d.get("cutoff_max", 10.0)),
        include_distance=_as_bool(d, "include_distance", True),
        include_histogram=_as_bool(d, "include_histogram", True),
        histogram_bins=int(d.get("histogram_bins", 10)),
        mode=str(d.get("mode", "simple")).strip(),
        cutoff_inclusive_min=_as_bool(d, "cutoff_inclusive_min", False),
        cutoff_inclusive_max=_as_bool(d, "cutoff_inclusive_max", False),
        scale=str(d.get("scale", "minmax_sklearn")).strip(),
    )


def load_topo_config(path: str) -> TopoConfig:
    d = load_json(path)
    sets = d.get("element_sets")
    return TopoConfig(
        neighbor_distance=float(d.get("neighbor_distance", 8.0)),
        element_sets=sets if isinstance(sets, list) else None,
    )


def load_other_config(path: str) -> OtherConfig:
    d = load_json(path)
    return OtherConfig(
        overwrite_graphs=_as_bool(d, "overwrite_graphs", True),
        jobs=int(d.get("jobs", 4)),
    )
