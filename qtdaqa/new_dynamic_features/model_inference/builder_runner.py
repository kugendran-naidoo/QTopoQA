from __future__ import annotations

import copy
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import yaml

from qtdaqa.new_dynamic_features.common.feature_metadata import (
    GraphFeatureMetadata,
    load_graph_feature_metadata,
)
from qtdaqa.new_dynamic_features.graph_builder.lib.features_config import DEFAULT_FEATURES


@dataclass
class BuilderConfig:
    jobs: int = 4
    topology_dedup_sort: Optional[bool] = None
    dump_edges: Optional[bool] = None
    feature_config: Optional[Path] = None
    features: Dict[str, object] = field(default_factory=dict)

    def prepare_feature_config(self, work_dir: Path) -> Optional[Path]:
        """Return a features.yaml path, creating one when overrides are supplied."""
        if self.feature_config:
            return self.feature_config
        payload = self._compose_features_payload()
        if payload is None:
            payload = self._default_features()
        target_dir = work_dir / "builder_features"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / "features.generated.yaml"
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return output_path

    def _compose_features_payload(self) -> Optional[Dict[str, object]]:
        if self.features:
            payload = copy.deepcopy(self.features)
        else:
            payload = self._default_features()

        if self.dump_edges is not None:
            options = payload.setdefault("options", {})
            if not isinstance(options, dict):
                raise ValueError("builder.features options must be a mapping when provided.")
            options["edge_dump"] = bool(self.dump_edges)

        if self.topology_dedup_sort is not None:
            topology = payload.setdefault("topology", {})
            if not isinstance(topology, dict):
                raise ValueError("builder.features.topology must be a mapping when provided.")
            params = topology.setdefault("params", {})
            if not isinstance(params, dict):
                raise ValueError("builder.features.topology.params must be a mapping when provided.")
            params["dedup_sort"] = bool(self.topology_dedup_sort)

        return payload

    @staticmethod
    def _default_features() -> Dict[str, object]:
        payload: Dict[str, object] = {}
        for key, entry in DEFAULT_FEATURES.items():
            payload[key] = {
                "module": entry.get("module"),
                "params": copy.deepcopy(entry.get("params", {})),
            }
        payload["options"] = {}
        return payload


def parse_builder_config(raw: object, base_dir: Path) -> BuilderConfig:
    if raw is None:
        return BuilderConfig()
    if not isinstance(raw, dict):
        raise ValueError("builder section must be a mapping when provided.")

    legacy_keys = {"topology_dedup_sort", "dump_edges", "features", "feature_config"}
    disallowed = sorted(legacy_keys.intersection(raw.keys()))
    if disallowed:
        raise ValueError(
            "builder configuration no longer accepts manual feature overrides "
            f"(disallowed keys: {', '.join(disallowed)}). Training metadata is used instead."
        )

    jobs = raw.get("jobs", 4)
    try:
        jobs_int = int(jobs)
    except (TypeError, ValueError) as exc:
        raise ValueError("builder.jobs must be an integer.") from exc
    jobs_int = max(1, jobs_int)

    return BuilderConfig(
        jobs=jobs_int,
    )


def _build_metadata_feature_payload(feature_metadata: Any) -> Optional[Dict[str, object]]:
    edge_schema: Optional[Dict[str, object]] = None
    if isinstance(feature_metadata, GraphFeatureMetadata):
        modules = feature_metadata.module_registry
        edge_schema = feature_metadata.edge_schema
    elif isinstance(feature_metadata, dict):
        modules = feature_metadata.get("module_registry")
        raw_edge_schema = feature_metadata.get("edge_schema")
        if isinstance(raw_edge_schema, dict):
            edge_schema = raw_edge_schema
    else:
        return None

    if not isinstance(modules, dict):
        return None

    payload: Dict[str, object] = {}
    for stage in ("interface", "topology", "node", "edge"):
        entry = modules.get(stage)
        if not isinstance(entry, dict):
            return None
        module_id = entry.get("id") or entry.get("module")
        if not module_id:
            return None

        stage_block: Dict[str, object] = {"module": module_id}
        alias = entry.get("alias")
        if alias:
            stage_block["alias"] = alias

        params: Optional[Dict[str, object]] = None
        entry_params = entry.get("params")
        if isinstance(entry_params, dict):
            params = copy.deepcopy(entry_params)
        defaults = entry.get("defaults")
        if isinstance(defaults, dict):
            if params is None:
                params = copy.deepcopy(defaults)
            else:
                params = {**copy.deepcopy(defaults), **params}

        if stage == "edge":
            schema_params = None
            if isinstance(edge_schema, dict):
                schema_params = edge_schema.get("module_params")
            if isinstance(schema_params, dict):
                if params is None:
                    params = copy.deepcopy(schema_params)
                else:
                    params = {**params, **copy.deepcopy(schema_params)}

        if isinstance(params, dict) and params:
            stage_block["params"] = params
        payload[stage] = stage_block

    payload["options"] = {}
    return payload


def _write_metadata_feature_config(
    feature_metadata: Any,
    work_dir: Path,
    fallback_metadata_path: Optional[Path] = None,
) -> Optional[Path]:
    payload = _build_metadata_feature_payload(feature_metadata)
    if payload is None and fallback_metadata_path is not None:
        try:
            raw = json.loads(fallback_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Unable to read graph metadata from %s: %s", fallback_metadata_path, exc)
        else:
            payload = _build_metadata_feature_payload(raw)
    if not payload:
        return None
    target_dir = work_dir / "builder_features"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "features.from_metadata.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path


def _resolve_metadata_source(final_schema: Dict[str, Dict[str, object]]) -> Optional[Path]:
    edge_schema = final_schema.get("edge_schema") or {}
    source = edge_schema.get("source")
    if not source:
        return None
    try:
        metadata_path = Path(source).expanduser().resolve()
    except OSError:
        return None
    return metadata_path if metadata_path.exists() else None


def run_graph_builder(cfg, metadata_feature_config: Optional[Path] = None) -> Path:
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")
    graph_dir = cfg.work_dir / "graph_data"
    builder_work = cfg.work_dir / "builder_work"
    builder_logs = cfg.work_dir / "builder_logs"

    for path in (graph_dir, builder_work, builder_logs):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    feature_config_path = metadata_feature_config or cfg.builder.prepare_feature_config(cfg.work_dir)

    cmd = [
        sys.executable,
        "-m",
        "qtdaqa.new_dynamic_features.graph_builder.graph_builder",
        "--dataset-dir",
        str(cfg.data_dir),
        "--work-dir",
        str(builder_work),
        "--graph-dir",
        str(graph_dir),
        "--log-dir",
        str(builder_logs),
        "--jobs",
        str(cfg.builder.jobs),
    ]
    if feature_config_path is not None:
        cmd.extend(["--feature-config", str(feature_config_path)])
    logging.info("Running graph builder: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return graph_dir


def _graph_metadata_matches(graph_dir: Path, final_schema: Dict[str, Dict[str, object]]) -> Tuple[bool, str]:
    expected_edge = final_schema.get("edge_schema") or {}
    if not expected_edge:
        return True, ""
    try:
        metadata = load_graph_feature_metadata(graph_dir, max_pt_samples=0)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"metadata load failed ({exc})"

    actual_edge = metadata.edge_schema or {}
    for key, expected_value in expected_edge.items():
        actual_value = actual_edge.get(key)
        if actual_value != expected_value:
            return False, f"edge schema key '{key}' mismatch (expected {expected_value}, found {actual_value})"
    return True, ""


def validate_graph_metadata(
    graph_dir: Path, final_schema: Dict[str, Dict[str, object]]
) -> GraphFeatureMetadata:
    try:
        metadata = load_graph_feature_metadata(graph_dir, max_pt_samples=0)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Unable to load graph metadata from {graph_dir}: {exc}. Rebuild graphs before running inference."
        ) from exc

    expected_edge = final_schema.get("edge_schema") or {}
    observed_edge = metadata.edge_schema or {}

    mismatches: List[str] = []
    for key in ("module", "variant", "dim", "bands", "module_params"):
        if key in expected_edge:
            expected_value = expected_edge.get(key)
            actual_value = observed_edge.get(key)
            if actual_value != expected_value:
                mismatches.append(
                    f"edge_schema.{key}: expected {expected_value!r}, observed {actual_value!r}"
                )

    expected_topology = final_schema.get("topology_schema") or {}
    observed_topology = metadata.node_schema or {}
    if "dim" in expected_topology:
        expected_dim = expected_topology.get("dim")
        actual_dim = observed_topology.get("dim")
        if actual_dim != expected_dim:
            mismatches.append(
                f"topology/node dim: expected {expected_dim!r}, observed {actual_dim!r}"
            )

    if mismatches:
        source = metadata.metadata_path or str(graph_dir / "graph_metadata.json")
        details = "; ".join(mismatches)
        raise RuntimeError(
            "Graph metadata is incompatible with the checkpoint schema. "
            f"Mismatches: {details}. Metadata source: {source}. "
            "Regenerate graphs using the matching feature configuration before rerunning inference."
        )

    return metadata


def ensure_graph_dir(
    cfg,
    final_schema: Dict[str, Dict[str, object]],
    feature_metadata: Optional[Dict[str, object]] = None,
) -> Path:
    work_dir = Path(cfg.work_dir)
    graph_dir = work_dir / "graph_data"
    reuse = bool(getattr(cfg, "reuse_existing_graphs", False))

    metadata_source = _resolve_metadata_source(final_schema)
    metadata_feature_config = None
    if feature_metadata or metadata_source is not None:
        metadata_feature_config = _write_metadata_feature_config(
            feature_metadata,
            work_dir,
            fallback_metadata_path=metadata_source,
        )

    if reuse and graph_dir.exists() and any(graph_dir.glob("*.pt")):
        matches, reason = _graph_metadata_matches(graph_dir, final_schema)
        if matches:
            logging.info("Reusing existing graphs at %s (metadata matched checkpoint schema).", graph_dir)
            return graph_dir
        logging.warning(
            "Existing graphs at %s are incompatible with checkpoint metadata (%s); rebuilding.", graph_dir, reason
        )
    return run_graph_builder(cfg, metadata_feature_config=metadata_feature_config)
