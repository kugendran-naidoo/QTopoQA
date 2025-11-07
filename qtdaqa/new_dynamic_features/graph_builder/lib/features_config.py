from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

LOG = logging.getLogger(__name__)

DEFAULT_FEATURES = {
    "interface": {
        "module": "interface/polar_cutoff/v1",
        "alias": None,
        "params": {},
    },
    "topology": {
        "module": "topology/persistence_basic/v1",
        "alias": None,
        "params": {},
    },
    "node": {
        "module": "node/dssp_topo_merge/v1",
        "alias": None,
        "params": {},
    },
    "edge": {
        "module": "edge/multi_scale/v24",
        "alias": None,
        "params": {},
    },
}


@dataclass
class FeatureSelection:
    interface: Dict[str, Any]
    topology: Dict[str, Any]
    node: Dict[str, Any]
    edge: Dict[str, Any]
    options: Dict[str, Any]


def _combine(default: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = dict(default)
    if override:
        for key, value in override.items():
            if key == "params":
                params = dict(default.get("params", {}))
                params.update(value or {})
                merged["params"] = params
            else:
                merged[key] = value
    if "params" not in merged:
        merged["params"] = {}
    if "alias" not in merged:
        merged["alias"] = None
    return merged


def _resolve_alias_from_catalog(data: Mapping[str, Any], catalog_key: str, module_id: str) -> Optional[str]:
    entries = data.get(catalog_key)
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("module") == module_id:
                alias = entry.get("alias")
                if alias:
                    return str(alias)
    return None


def load_feature_config(config_path: Optional[Path]) -> FeatureSelection:
    data: Dict[str, Any] = {}
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
            if not isinstance(content, dict):
                raise ValueError(f"Feature config at {config_path} must be a mapping.")
            data = content
        _validate_schema(data, config_path)

    alias_map_raw = data.pop("aliases", {})
    defaults_raw = data.get("defaults", {})
    alias_map: Dict[str, str] = {}
    if alias_map_raw:
        if not isinstance(alias_map_raw, dict):
            raise ValueError("'aliases' section must be a mapping of module_id -> alias string.")
        for key, value in alias_map_raw.items():
            if value is not None:
                alias_map[str(key)] = str(value)

    default_jobs = None
    if isinstance(defaults_raw, dict):
        jobs_value = defaults_raw.get("jobs")
        if jobs_value is not None:
            try:
                jobs_int = int(jobs_value)
                if jobs_int > 0:
                    default_jobs = jobs_int
            except (TypeError, ValueError):
                raise ValueError("defaults.jobs must be an integer > 0 if specified.")

    selections = {
        key: _combine(DEFAULT_FEATURES[key], data.get(key))
        for key in ("interface", "topology", "node", "edge")
    }

    catalog_map = {
        "interface": "Interface modules",
        "topology": "Topology modules",
        "node": "Node modules",
        "edge": "Edge modules",
    }
    for key, catalog_key in catalog_map.items():
        if selections[key].get("alias") in (None, ""):
            module_id = selections[key].get("module")
            if module_id:
                alias = _resolve_alias_from_catalog(data, catalog_key, module_id)
                if not alias:
                    alias = alias_map.get(module_id)
                if alias:
                    selections[key]["alias"] = alias
    options = dict(data.get("options", {}))
    if default_jobs is not None:
        options.setdefault("default_jobs", default_jobs)

    return FeatureSelection(
        interface=selections["interface"],
        topology=selections["topology"],
        node=selections["node"],
        edge=selections["edge"],
        options=options,
    )


REQUIRED_SECTIONS = ("interface", "node", "edge")
META_SECTIONS = {"defaults", "aliases", "options"}


def _validate_schema(data: Mapping[str, Any], config_path: Path) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing:
        raise ValueError(
            f"Feature config {config_path} missing required sections: {', '.join(missing)}. "
            "Each run must specify interface/node/edge modules explicitly."
        )
    for section in REQUIRED_SECTIONS:
        entry = data.get(section)
        if not isinstance(entry, Mapping) or "module" not in entry:
            raise ValueError(
                f"Feature config {config_path} section '{section}' must provide a 'module' field."
            )

    for section, entry in data.items():
        if section in REQUIRED_SECTIONS or section in META_SECTIONS:
            continue
        if isinstance(entry, Mapping) and "module" in entry:
            continue  # optional, stage-like block
        raise ValueError(
            f"Unsupported section '{section}' in {config_path}. Remove catalog/template blocks and keep only concrete stage definitions."
        )
