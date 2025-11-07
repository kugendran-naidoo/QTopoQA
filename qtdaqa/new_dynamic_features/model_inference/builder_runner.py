from __future__ import annotations

import copy
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

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
            return None
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
            if self.dump_edges is None and self.topology_dedup_sort is None:
                return None
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


def run_graph_builder(cfg) -> Path:
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")
    graph_dir = cfg.work_dir / "graph_data"
    builder_work = cfg.work_dir / "builder_work"
    builder_logs = cfg.work_dir / "builder_logs"

    for path in (graph_dir, builder_work, builder_logs):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    feature_config_path = cfg.builder.prepare_feature_config(cfg.work_dir)

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


def ensure_graph_dir(cfg, final_schema: Dict[str, Dict[str, object]]) -> Path:
    work_dir = Path(cfg.work_dir)
    graph_dir = work_dir / "graph_data"
    reuse = bool(getattr(cfg, "reuse_existing_graphs", False))

    if reuse and graph_dir.exists() and any(graph_dir.glob("*.pt")):
        matches, reason = _graph_metadata_matches(graph_dir, final_schema)
        if matches:
            logging.info("Reusing existing graphs at %s (metadata matched checkpoint schema).", graph_dir)
            return graph_dir
        logging.warning(
            "Existing graphs at %s are incompatible with checkpoint metadata (%s); rebuilding.", graph_dir, reason
        )
    return run_graph_builder(cfg)
