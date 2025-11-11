from __future__ import annotations

from typing import Any, Dict

from qtdaqa.new_dynamic_features.graph_builder2.modules.base import (
    EdgeFeatureModule,
    InterfaceFeatureModule,
    NodeFeatureModule,
    TopologyFeatureModule,
    build_metadata,
    ensure_sorted_float_sequence,
    require_bool,
    require_float,
    require_int,
    require_positive_float,
    require_positive_int,
)
from qtdaqa.new_dynamic_features.graph_builder2.modules.registry import register_feature_module


class _InterfaceStub(InterfaceFeatureModule):
    module_id = "interface/polar_cutoff/v1"
    module_kind = "interface"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="stub interface",
        description="stub",
    )

    def extract_interfaces(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        cutoff = params.get("cutoff")
        if cutoff is not None:
            params["cutoff"] = require_positive_float(cutoff, "interface.params.cutoff")
        decimals = params.get("coordinate_decimals")
        if decimals is not None:
            value = require_int(decimals, "interface.params.coordinate_decimals")
            if value < -1:
                raise ValueError("interface.params.coordinate_decimals must be >= -1.")
            params["coordinate_decimals"] = value
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "interface.params.jobs")


class _TopologyStub(TopologyFeatureModule):
    module_id = "topology/persistence_basic/v1"
    module_kind = "topology"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="stub topology",
        description="stub",
    )

    def generate_topology(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

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
        filters = params.get("element_filters")
        if filters is not None:
            if isinstance(filters, str) or not isinstance(filters, (list, tuple)):
                raise ValueError("topology.params.element_filters must be a list/tuple.")
            normalised = []
            for entry in filters:
                if not isinstance(entry, (list, tuple)) or not entry:
                    raise ValueError("Each element_filters entry must be a non-empty list/tuple.")
                cleaned = []
                for symbol in entry:
                    if not isinstance(symbol, str) or not symbol.strip():
                        raise ValueError("Element filter symbols must be non-empty strings.")
                    cleaned.append(symbol.strip())
                normalised.append(tuple(cleaned))
            params["element_filters"] = tuple(normalised)


class _NodeStub(NodeFeatureModule):
    module_id = "node/dssp_topo_merge/v1"
    module_kind = "node"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="stub node",
        description="stub",
    )

    def generate_nodes(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        drop_na = params.get("drop_na")
        if drop_na is not None:
            params["drop_na"] = require_bool(drop_na, "node.params.drop_na")
        jobs = params.get("jobs")
        if jobs is not None:
            params["jobs"] = require_positive_int(jobs, "node.params.jobs")


class _LegacyEdgeStub(EdgeFeatureModule):
    module_id = "edge/legacy_band/v11"
    module_kind = "edge"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="stub edge legacy",
        description="stub",
    )

    def build_edges(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        distance_min = params.get("distance_min")
        distance_max = params.get("distance_max")
        if distance_min is not None:
            params["distance_min"] = require_float(distance_min, "edge.params.distance_min")
        if distance_max is not None:
            params["distance_max"] = require_float(distance_max, "edge.params.distance_max")
        if distance_min is not None and distance_max is not None:
            if params["distance_min"] >= params["distance_max"]:
                raise ValueError("edge.params.distance_min must be < distance_max.")
        scale_features = params.get("scale_features")
        if scale_features is not None:
            params["scale_features"] = require_bool(scale_features, "edge.params.scale_features")


class _MultiEdgeStub(EdgeFeatureModule):
    module_id = "edge/multi_scale/v24"
    module_kind = "edge"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="stub edge multi",
        description="stub",
    )

    def build_edges(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> None:
        histogram_bins = params.get("histogram_bins")
        if histogram_bins is not None:
            params["histogram_bins"] = list(
                ensure_sorted_float_sequence(histogram_bins, "edge.params.histogram_bins")
            )


class _MolEdgeStub(EdgeFeatureModule):
    module_id = "custom/mol_stage/v1"
    module_kind = "mol"
    _metadata = build_metadata(
        module_id=module_id,
        module_kind=module_kind,
        summary="stub mol",
        description="stub",
    )

    def build_edges(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _register(cls):
    try:
        register_feature_module(cls)
    except ValueError:
        pass


for stub_cls in (
    _InterfaceStub,
    _TopologyStub,
    _NodeStub,
    _LegacyEdgeStub,
    _MultiEdgeStub,
    _MolEdgeStub,
):
    _register(stub_cls)
