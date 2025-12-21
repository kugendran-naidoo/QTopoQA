from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from qtdaqa.new_dynamic_features.common.feature_metadata import (
    GraphFeatureMetadata,
    load_graph_feature_metadata,
)
from qtdaqa.new_dynamic_features.graph_builder2.lib.features_config import DEFAULT_FEATURES
from qtdaqa.new_dynamic_features.graph_builder2.modules import instantiate_module

GRAPH_BUILDER2_MODULE = "qtdaqa.new_dynamic_features.graph_builder2.graph_builder2"
# Legacy alias retained for compatibility with tests/configs
LEGACY_BUILDER_MODULE = GRAPH_BUILDER2_MODULE
_BUILDER_MODULES = {
    "graph_builder2": GRAPH_BUILDER2_MODULE,
}


@dataclass
class BuilderConfig:
    jobs: int = 4
    topology_dedup_sort: Optional[bool] = None
    dump_edges: Optional[bool] = None
    sort_artifacts: Optional[bool] = None
    feature_config: Optional[Path] = None
    features: Dict[str, object] = field(default_factory=dict)
    builder_name: Optional[str] = None

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


def _resolve_optional_path(path_raw: Optional[object], base_dir: Path) -> Optional[Path]:
    if path_raw is None:
        return None
    if isinstance(path_raw, Path):
        candidate = path_raw
    else:
        candidate = Path(str(path_raw))
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def parse_builder_config(raw: object, base_dir: Path) -> BuilderConfig:
    if raw is None:
        return BuilderConfig()
    if not isinstance(raw, dict):
        raise ValueError("builder section must be a mapping when provided.")

    legacy_keys = {"topology_dedup_sort", "dump_edges", "features"}
    disallowed = sorted(legacy_keys.intersection(raw.keys()))
    if disallowed:
        raise ValueError(
            "builder configuration no longer accepts manual feature overrides "
            f"(disallowed keys: {', '.join(disallowed)}). Training metadata is used instead."
        )

    allowed_keys = {"jobs", "id", "name", "builder", "sort_artifacts", "feature_config"}
    unknown = sorted(key for key in raw.keys() if key not in allowed_keys and key not in legacy_keys)
    if unknown:
        raise ValueError(
            "builder configuration has unrecognized keys: "
            f"{', '.join(unknown)}. Remove typos/unknown fields."
        )

    jobs = raw.get("jobs", 4)
    try:
        jobs_int = int(jobs)
    except (TypeError, ValueError) as exc:
        raise ValueError("builder.jobs must be an integer.") from exc
    jobs_int = max(1, jobs_int)

    builder_name_raw = raw.get("id") or raw.get("name") or raw.get("builder")
    builder_name = str(builder_name_raw).strip() if builder_name_raw else None

    sort_artifacts_raw = raw.get("sort_artifacts")
    if sort_artifacts_raw is None:
        sort_artifacts_flag = None
    elif isinstance(sort_artifacts_raw, bool):
        sort_artifacts_flag = sort_artifacts_raw
    else:
        raise ValueError("builder.sort_artifacts must be a boolean when provided.")

    feature_config_path = _resolve_optional_path(raw.get("feature_config"), base_dir)
    if feature_config_path is not None and not feature_config_path.exists():
        raise FileNotFoundError(f"builder.feature_config not found: {feature_config_path}")

    return BuilderConfig(
        jobs=jobs_int,
        builder_name=builder_name or None,
        sort_artifacts=sort_artifacts_flag,
        feature_config=feature_config_path,
    )


_MISSING = object()


def _extract_module_defaults(module_registry: Optional[Dict[str, object]], module_id: Optional[str]) -> Dict[str, object]:
    """Best-effort lookup of defaults for a module id from module_registry."""
    if not module_registry or not module_id:
        return {}
    if not isinstance(module_registry, dict):
        return {}
    candidates: List[Dict[str, object]] = []
    edge_entry = module_registry.get("edge")
    if isinstance(edge_entry, dict):
        candidates.append(edge_entry)
    for entry in module_registry.values():
        if isinstance(entry, dict):
            candidates.append(entry)
    for entry in candidates:
        module_value = entry.get("id") or entry.get("module")
        if module_value != module_id:
            continue
        defaults: Dict[str, object] = {}
        entry_defaults = entry.get("defaults")
        if isinstance(entry_defaults, dict):
            defaults.update(entry_defaults)
        entry_params = entry.get("params")
        if isinstance(entry_params, dict):
            # params recorded in module_registry act as defaults for comparison purposes
            defaults.update(entry_params)
        return copy.deepcopy(defaults)
    return {}


def _normalize_params(params: Optional[Dict[str, object]], defaults: Optional[Dict[str, object]]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    if defaults:
        merged.update(copy.deepcopy(defaults))
    if params:
        merged.update(params)
    return merged


def _params_equivalent(expected: Optional[Dict[str, object]], observed: Optional[Dict[str, object]]) -> Tuple[bool, str]:
    """Treat missing keys as default/falsey; require equality otherwise."""
    exp = expected or {}
    obs = observed or {}
    keys = set(exp.keys()) | set(obs.keys())
    for key in keys:
        v_exp = exp.get(key, _MISSING)
        v_obs = obs.get(key, _MISSING)
        if v_exp is _MISSING and v_obs in (False, None, 0, "", [], {}):
            continue
        if v_obs is _MISSING and v_exp in (False, None, 0, "", [], {}):
            continue
        if v_exp == v_obs:
            continue
        return False, f"key '{key}' mismatch (expected {v_exp!r}, found {v_obs!r})"
    return True, ""


def _build_metadata_feature_payload(feature_metadata: Any) -> Optional[Dict[str, object]]:
    # If a full feature-config snapshot is available in the checkpoint metadata, prefer it.
    builder_block: Optional[Dict[str, object]] = None
    feature_config_snapshot: Optional[Dict[str, object]] = None
    if isinstance(feature_metadata, GraphFeatureMetadata):
        builder_block = feature_metadata.builder if isinstance(feature_metadata.builder, dict) else None
        if isinstance(feature_metadata.feature_config, dict):
            feature_config_snapshot = feature_metadata.feature_config
    elif isinstance(feature_metadata, dict):
        builder_block = feature_metadata.get("builder") if isinstance(feature_metadata.get("builder"), dict) else None
        if isinstance(feature_metadata.get("feature_config"), dict):
            feature_config_snapshot = feature_metadata.get("feature_config")
    if builder_block and not feature_config_snapshot:
        fc = builder_block.get("feature_config")
        if isinstance(fc, dict):
            feature_config_snapshot = fc

    def _payload_from_snapshot(snapshot: Dict[str, object]) -> Optional[Dict[str, object]]:
        payload = None
        text = snapshot.get("text")
        if isinstance(text, str) and text.strip():
            try:
                payload = yaml.safe_load(text) or {}
            except Exception:
                payload = None
        if payload is None:
            path_val = snapshot.get("path")
            if path_val:
                try:
                    payload = yaml.safe_load(Path(str(path_val)).read_text()) or {}
                except Exception:
                    payload = None
        if payload is None:
            return None
        # Merge builder options if present
        options: Dict[str, object] = {}
        if isinstance(builder_block, dict):
            builder_opts = builder_block.get("options")
            if isinstance(builder_opts, dict):
                options.update(builder_opts)
        if options:
            payload.setdefault("options", {}).update(options)
        return payload

    if isinstance(feature_config_snapshot, dict):
        payload = _payload_from_snapshot(feature_config_snapshot)
        if payload:
            return payload

    edge_schema: Optional[Dict[str, object]] = None
    if isinstance(feature_metadata, GraphFeatureMetadata):
        modules = feature_metadata.module_registry
        edge_schema = feature_metadata.edge_schema
        metadata_path = getattr(feature_metadata, "metadata_path", None)
    elif isinstance(feature_metadata, dict):
        modules = feature_metadata.get("module_registry")
        raw_edge_schema = feature_metadata.get("edge_schema")
        if isinstance(raw_edge_schema, dict):
            edge_schema = raw_edge_schema
        metadata_path = feature_metadata.get("metadata_path")
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

    # Attempt to populate options from builder block or underlying graph metadata
    builder_block = _extract_builder_block(feature_metadata)
    options_block: Dict[str, object] = {}
    if isinstance(builder_block, dict):
        builder_options = builder_block.get("options")
        if isinstance(builder_options, dict):
            options_block.update(copy.deepcopy(builder_options))
    if not options_block and metadata_path:
        try:
            raw = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            builder_meta = raw.get("_builder") if isinstance(raw, dict) else None
            if isinstance(builder_meta, dict):
                builder_options = builder_meta.get("options")
                if isinstance(builder_options, dict):
                    options_block.update(builder_options)
        except Exception:
            pass

    payload["options"] = options_block
    return payload


def _write_feature_config_payload(work_dir: Path, payload: Dict[str, object], filename: str) -> Path:
    target_dir = work_dir / "builder_features"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path


def _extract_builder_block(source: Any) -> Optional[Dict[str, object]]:
    if isinstance(source, GraphFeatureMetadata):
        return source.builder
    if isinstance(source, dict):
        block = source.get("builder")
        if isinstance(block, dict):
            return block
    return None


def _infer_builder_id(builder_block: Optional[Dict[str, object]]) -> Optional[str]:
    if not isinstance(builder_block, dict):
        return None
    builder_id = builder_block.get("id")
    if isinstance(builder_id, str) and builder_id.strip():
        return builder_id.strip()
    module_name = builder_block.get("module")
    if isinstance(module_name, str):
        if "graph_builder2" in module_name:
            return "graph_builder2"
        if "graph_builder" in module_name:
            return "graph_builder"
    return None


def _resolve_builder_module(identifier: Optional[str]) -> Optional[str]:
    if identifier is None:
        return None
    token = identifier.strip()
    key = token.lower()
    if key in _BUILDER_MODULES:
        return _BUILDER_MODULES[key]
    if token.startswith("qtdaqa."):
        return token
    return None


def _determine_builder_choice(cfg, feature_metadata: Any) -> Tuple[str, str]:
    candidates = [
        ("environment variable", os.environ.get("QTOPO_FORCE_BUILDER")),
        ("config", getattr(cfg.builder, "builder_name", None)),
        ("checkpoint", _infer_builder_id(_extract_builder_block(feature_metadata))),
    ]
    for source, value in candidates:
        module = _resolve_builder_module(value)
        if module:
            return module, source
        if value:
            logging.warning("Unknown builder override '%s' from %s; falling back to defaults.", value, source)
    return GRAPH_BUILDER2_MODULE, "default"


def _write_builder_snapshot_config(
    builder_block: Optional[Dict[str, object]],
    work_dir: Path,
) -> Optional[Path]:
    if not isinstance(builder_block, dict):
        return None
    snapshot = builder_block.get("feature_config")
    if not isinstance(snapshot, dict):
        return None

    payload: Optional[Dict[str, object]] = None
    text = snapshot.get("text")
    if isinstance(text, str) and text.strip():
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            logging.warning("Unable to parse inline feature-config snapshot: %s", exc)
            payload = None

    if payload is None:
        path_value = snapshot.get("path")
        if isinstance(path_value, str):
            snapshot_path = Path(path_value).expanduser()
            if snapshot_path.exists():
                try:
                    payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError) as exc:
                    logging.warning("Unable to load feature-config snapshot from %s: %s", snapshot_path, exc)
            else:
                logging.warning("Feature-config snapshot path %s not found.", snapshot_path)

    if not isinstance(payload, dict):
        return None
    return _write_feature_config_payload(work_dir, payload, "features.from_checkpoint.yaml")


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


def _write_metadata_feature_config(
    feature_metadata: Any,
    work_dir: Path,
    fallback_metadata_path: Optional[Path] = None,
) -> Optional[Path]:
    payload = _build_metadata_feature_payload(feature_metadata)
    if payload:
        if not payload.get("options"):
            logging.warning(
                "Builder options were not available in checkpoint metadata; generated config will omit them."
            )
        return _write_feature_config_payload(work_dir, payload, "features.from_metadata.yaml")

    snapshot_path = _write_builder_snapshot_config(_extract_builder_block(feature_metadata), work_dir)
    if snapshot_path:
        return snapshot_path

    fallback_payload: Optional[Dict[str, object]] = None
    fallback_builder: Optional[Dict[str, object]] = None
    if fallback_metadata_path is not None:
        try:
            raw = json.loads(fallback_metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Unable to read graph metadata from %s: %s", fallback_metadata_path, exc)
        else:
            fallback_payload = _build_metadata_feature_payload(raw)
            fallback_builder = _extract_builder_block(raw)

    if fallback_payload:
        return _write_feature_config_payload(work_dir, fallback_payload, "features.from_metadata.yaml")

    snapshot_path = _write_builder_snapshot_config(fallback_builder, work_dir)
    if snapshot_path:
        return snapshot_path

    return None


def _build_schema_feature_payload(final_schema: Dict[str, Dict[str, object]]) -> Optional[Dict[str, object]]:
    edge_schema = final_schema.get("edge_schema")
    if not isinstance(edge_schema, dict):
        return None
    payload: Dict[str, object] = {}
    for key, entry in DEFAULT_FEATURES.items():
        payload[key] = {
            "module": entry.get("module"),
            "params": copy.deepcopy(entry.get("params", {})),
        }
    payload["options"] = {}

    def _apply(stage_key: str, schema_key: str) -> None:
        schema = final_schema.get(schema_key)
        if not isinstance(schema, dict) or not schema:
            return
        block = payload.get(stage_key, {})
        module_id = schema.get("module")
        if module_id:
            block["module"] = module_id
        module_params = schema.get("module_params")
        if isinstance(module_params, dict):
            block["params"] = copy.deepcopy(module_params)
        if "jobs" in schema:
            block["jobs"] = schema.get("jobs")
        payload[stage_key] = block

    _apply("interface", "interface_schema")
    _apply("topology", "topology_schema")
    _apply("node", "node_schema")
    _apply("edge", "edge_schema")
    return payload


def _preflight_registry_support(final_schema: Dict[str, Dict[str, object]]) -> None:
    """Ensure the requested modules/variants are available locally before spawning builder."""
    edge_schema = final_schema.get("edge_schema") or {}
    module_id = edge_schema.get("module")
    if not module_id:
        return
    params = edge_schema.get("module_params") or {}
    try:
        instantiate_module(module_id, **params)
    except KeyError as exc:
        logging.warning(
            "Requested edge module '%s' is not registered locally; skipping preflight check. "
            "Ensure graph_builder2 with this module is available at runtime.",
            module_id,
        )
        return
    except Exception as exc:
        raise RuntimeError(
            f"Edge module '{module_id}' rejected parameters {params}: {exc}. "
            "Verify feature schema matches the local builder version."
        ) from exc


def run_graph_builder(
    cfg,
    metadata_feature_config: Optional[Path] = None,
    builder_module: Optional[str] = None,
) -> Path:
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

    module_path = builder_module or GRAPH_BUILDER2_MODULE

    cmd = [
        sys.executable,
        "-m",
        module_path,
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
    if cfg.builder.sort_artifacts is False:
        cmd.append("--no-sort-artifacts")
    if feature_config_path is not None:
        cmd.extend(["--feature-config", str(feature_config_path)])
    logging.info("Running graph builder (%s): %s", module_path, " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        snippet = _summarize_builder_logs(builder_logs)
        raise RuntimeError(
            f"Graph builder failed with return code {exc.returncode}. "
            f"Logs: {builder_logs}. Command: {' '.join(cmd)}"
            + (f"\nLast logs:\n{snippet}" if snippet else "")
        ) from exc
    return graph_dir


def _graph_metadata_matches(graph_dir: Path, final_schema: Dict[str, Dict[str, object]]) -> Tuple[bool, str]:
    expected_edge = final_schema.get("edge_schema") or {}
    expected_node = final_schema.get("node_schema") or {}
    expected_topology = final_schema.get("topology_schema") or {}
    if not expected_edge:
        return True, ""
    try:
        metadata = load_graph_feature_metadata(graph_dir, max_pt_samples=0)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"metadata load failed ({exc})"

    actual_edge = metadata.edge_schema or {}
    actual_node = getattr(metadata, "node_schema", {}) or {}
    defaults = _extract_module_defaults(
        getattr(metadata, "module_registry", {}) or {},
        expected_edge.get("module") or actual_edge.get("module"),
    )
    for key, expected_value in expected_edge.items():
        actual_value = actual_edge.get(key)
        if key == "module_params":
            exp_params = _normalize_params(expected_value if isinstance(expected_value, dict) else {}, defaults)
            obs_params = _normalize_params(actual_value if isinstance(actual_value, dict) else {}, defaults)
            ok, reason = _params_equivalent(exp_params, obs_params)
            if not ok:
                return False, f"edge schema module_params mismatch: {reason}"
            continue
        if actual_value != expected_value:
            return False, f"edge schema key '{key}' mismatch (expected {expected_value}, found {actual_value})"
    if expected_node:
        exp_node_dim = expected_node.get("dim")
        if exp_node_dim is not None:
            try:
                exp_node_dim = int(exp_node_dim)
            except (TypeError, ValueError):
                return False, f"node schema dim is not an integer: {exp_node_dim!r}"
            obs_node_dim = actual_node.get("dim")
            if obs_node_dim is None:
                return False, "node schema dim missing in graph metadata"
            try:
                obs_node_dim_int = int(obs_node_dim)
            except (TypeError, ValueError):
                return False, f"node schema dim in graph metadata is not an integer: {obs_node_dim!r}"
            if obs_node_dim_int != exp_node_dim:
                return False, f"node schema dim mismatch (expected {exp_node_dim}, found {obs_node_dim_int})"
    if expected_topology:
        exp_topo_dim = expected_topology.get("dim")
        if exp_topo_dim is not None:
            try:
                exp_topo_dim = int(exp_topo_dim)
            except (TypeError, ValueError):
                return False, f"topology schema dim is not an integer: {exp_topo_dim!r}"
            observed_topology = getattr(metadata, "topology_schema", {}) or {}
            obs_topo_dim = observed_topology.get("dim")
            if obs_topo_dim is None:
                # fall back to spec if present
                obs_topo_dim = (getattr(metadata, "topology_schema_spec", {}) or {}).get("dim")
            if obs_topo_dim is None:
                return False, "topology schema dim missing in graph metadata"
            try:
                obs_topo_dim_int = int(obs_topo_dim)
            except (TypeError, ValueError):
                return False, f"topology schema dim in graph metadata is not an integer: {obs_topo_dim!r}"
            if obs_topo_dim_int != exp_topo_dim:
                return False, f"topology schema dim mismatch (expected {exp_topo_dim}, found {obs_topo_dim_int})"
    return True, ""


def _summarize_builder_logs(log_dir: Path, tail_lines: int = 40) -> str:
    """Return a small tail from the most recent builder log, if available."""
    if not log_dir.exists():
        return ""
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        return ""
    latest = log_files[0]
    try:
        lines = latest.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-tail_lines:])


def validate_graph_metadata(
    graph_dir: Path, final_schema: Dict[str, Dict[str, object]]
) -> GraphFeatureMetadata:
    # Allow overriding strict validation for reuse-only scenarios.
    if os.environ.get("QTOPO_REUSE_ONLY", "").strip().lower() in {"1", "true", "yes"}:
        return load_graph_feature_metadata(graph_dir, max_pt_samples=0)
    try:
        metadata = load_graph_feature_metadata(graph_dir, max_pt_samples=0)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Unable to load graph metadata from {graph_dir}: {exc}. Rebuild graphs before running inference."
        ) from exc

    expected_edge = final_schema.get("edge_schema") or {}
    observed_edge = metadata.edge_schema or {}
    defaults = _extract_module_defaults(
        getattr(metadata, "module_registry", {}) or {},
        expected_edge.get("module") or observed_edge.get("module"),
    )

    mismatches: List[str] = []
    for key in ("module", "variant", "dim", "bands", "module_params"):
        if key in expected_edge:
            expected_value = expected_edge.get(key)
            actual_value = observed_edge.get(key)
            if key == "module_params":
                exp_params = _normalize_params(expected_value if isinstance(expected_value, dict) else {}, defaults)
                obs_params = _normalize_params(actual_value if isinstance(actual_value, dict) else {}, defaults)
                ok, reason = _params_equivalent(exp_params, obs_params)
                if not ok:
                    mismatches.append(f"edge_schema.module_params: {reason}")
                continue
            if actual_value != expected_value:
                mismatches.append(
                    f"edge_schema.{key}: expected {expected_value!r}, observed {actual_value!r}"
                )

    expected_node = final_schema.get("node_schema") or {}
    observed_node = metadata.node_schema or {}
    if "dim" in expected_node:
        expected_dim = expected_node.get("dim")
        actual_dim = observed_node.get("dim")
        if actual_dim != expected_dim:
            mismatches.append(
                f"node dim: expected {expected_dim!r}, observed {actual_dim!r}"
            )
    if "module" in expected_node:
        expected_node_module = expected_node.get("module")
        actual_node_module = observed_node.get("module")
        if actual_node_module != expected_node_module:
            mismatches.append(
                f"node_schema.module: expected {expected_node_module!r}, observed {actual_node_module!r}"
            )

    expected_topology = final_schema.get("topology_schema") or {}
    observed_topology = getattr(metadata, "topology_schema", {}) or {}
    if "dim" in expected_topology:
        expected_dim_raw = expected_topology.get("dim")
        actual_dim_raw = observed_topology.get("dim") or (getattr(metadata, "topology_schema_spec", {}) or {}).get("dim")
        try:
            expected_dim_int = int(expected_dim_raw) if expected_dim_raw is not None else None
        except (TypeError, ValueError):
            mismatches.append(f"topology_schema.dim is not an integer: {expected_dim_raw!r}")
            expected_dim_int = None
        try:
            actual_dim_int = int(actual_dim_raw) if actual_dim_raw is not None else None
        except (TypeError, ValueError):
            mismatches.append(f"topology_schema.dim in graph metadata is not an integer: {actual_dim_raw!r}")
            actual_dim_int = None
        if expected_dim_int is not None and actual_dim_int is None:
            mismatches.append("topology dim missing in graph metadata")
        elif expected_dim_int is not None and actual_dim_int is not None and actual_dim_int != expected_dim_int:
            mismatches.append(
                f"topology dim: expected {expected_dim_int!r}, observed {actual_dim_int!r}"
            )
    if "module" in expected_topology:
        expected_topology_module = expected_topology.get("module")
        actual_topology_module = observed_topology.get("module")
        if actual_topology_module != expected_topology_module:
            mismatches.append(
                f"topology_schema.module: expected {expected_topology_module!r}, observed {actual_topology_module!r}"
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
    # Fail fast if the requested module/variant isn't available locally.
    _preflight_registry_support(final_schema)

    work_dir = Path(cfg.work_dir)
    graph_dir = work_dir / "graph_data"
    reuse = bool(getattr(cfg, "reuse_existing_graphs", False))
    builder_module, builder_source = _determine_builder_choice(cfg, feature_metadata)

    metadata_source = _resolve_metadata_source(final_schema)
    metadata_feature_config = None
    if feature_metadata or metadata_source is not None:
        metadata_feature_config = _write_metadata_feature_config(
            feature_metadata,
            work_dir,
            fallback_metadata_path=metadata_source,
        )
    if metadata_feature_config is None:
        if os.environ.get("QTOPO_STRICT_BUILDER", "").strip().lower() in {"1", "true", "yes"}:
            raise RuntimeError("Checkpoint metadata lacked feature_config snapshot; strict builder mode is enabled.")
        schema_payload = _build_schema_feature_payload(final_schema)
        if schema_payload:
            metadata_feature_config = _write_feature_config_payload(
                work_dir, schema_payload, "features.from_schema.yaml"
            )

    if reuse and graph_dir.exists() and any(graph_dir.rglob("*.pt")):
        matches, reason = _graph_metadata_matches(graph_dir, final_schema)
        if matches:
            logging.info("Reusing existing graphs at %s (metadata matched checkpoint schema).", graph_dir)
            return graph_dir
        if os.environ.get("QTOPO_REUSE_ONLY"):
            logging.warning(
                "Reuse-only mode enabled; keeping existing graphs at %s despite metadata mismatch (%s).",
                graph_dir,
                reason,
            )
            return graph_dir
        logging.warning(
            "Existing graphs at %s are incompatible with checkpoint metadata (%s); rebuilding.", graph_dir, reason
        )
    logging.info("Building graphs using %s (selection source: %s).", builder_module, builder_source)
    return run_graph_builder(
        cfg,
        metadata_feature_config=metadata_feature_config,
        builder_module=builder_module,
    )
