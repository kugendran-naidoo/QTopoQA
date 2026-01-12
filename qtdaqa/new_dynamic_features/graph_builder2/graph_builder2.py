#!/usr/bin/env python3
"""
Dynamic graph builder with pluggable feature modules.

Stages
------
1. Interface extraction
2. Topological feature generation
3. Node feature construction
4. Graph assembly (.pt) with configurable edge features

All feature stages are controlled via ``features.yaml`` which maps each feature
kind (interface/topology/node/edge) to a registered module identifier and
parameter overrides. New modules can be dropped into ``modules/`` and registered
with ``register_feature_module`` for automatic discovery.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Type

import yaml

try:
    from .lib.features_config import FeatureSelection, load_feature_config
    from .lib.directory_permissions import ensure_tree_readable, ensure_tree_readwrite
    from .lib.log_dirs import LogDirectoryInfo, prepare_log_directory
    from .lib.stage_common import index_structures
    from .lib.pdb_utils import configure_pdb_parser
    from .builder_info import build_builder_info
except ImportError:  # pragma: no cover - fallback for direct execution
    from lib.features_config import FeatureSelection, load_feature_config
    from lib.directory_permissions import ensure_tree_readable, ensure_tree_readwrite
    from lib.log_dirs import LogDirectoryInfo, prepare_log_directory
    from lib.stage_common import index_structures
    from lib.pdb_utils import configure_pdb_parser  # type: ignore
    from builder_info import build_builder_info

try:  # Optional dependency for post-graph validation/manifest writing.
    from qtdaqa.new_dynamic_features.model_training2.tools.new_validate_graphs import (
        validate as validate_graphs,
    )
except Exception:  # pragma: no cover - allow running without model_training2 tools
    validate_graphs = None

if TYPE_CHECKING:  # pragma: no cover
    from .modules.base import (
        EdgeFeatureModule,
        InterfaceFeatureModule,
        NodeFeatureModule,
        TopologyFeatureModule,
    )
else:  # pragma: no cover - runtime placeholders
    EdgeFeatureModule = InterfaceFeatureModule = NodeFeatureModule = TopologyFeatureModule = object


LOG = logging.getLogger("graph_builder")

_STAGE_ORDER = ("interface", "topology", "node", "edge")
_REQUIRED_STAGES = {"interface", "node", "edge"}
_TEMPLATE_DEFAULTS = {
    "interface": "interface/polar_cutoff/v1",
    "topology": "topology/persistence_basic/v1",
    "node": "node/dssp_topo_merge/v1",
    "edge": "edge/legacy_band/v11",
}
_ROUNDING_SENTINEL = -1
_ROUNDING_MAX = 15

def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _format_module(module_id: str, alias: Optional[str]) -> str:
    return f"{module_id} [alias: {alias}]" if alias else module_id


def _resolve_jobs(module, fallback: Optional[int] = None) -> Optional[int]:
    params = getattr(module, "params", None)
    if isinstance(params, dict) and "jobs" in params:
        value = params.get("jobs")
        if value in (None, "auto"):
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback
    return fallback


def _resolve_edge_dump(cli_flag: Optional[bool], config_value: Optional[object]) -> bool:
    if cli_flag is not None:
        return bool(cli_flag)
    if isinstance(config_value, bool):
        return config_value
    return True


def _resolve_edge_dump_dir(work_dir: Path, cli_dir: Optional[Path], config_dir: Optional[object]) -> Path:
    if cli_dir is not None:
        candidate = cli_dir
    elif isinstance(config_dir, str) and config_dir.strip():
        candidate = Path(config_dir)
    else:
        candidate = work_dir / "edge_features"
    return candidate.expanduser().resolve()


def _resolve_round_decimals(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    try:
        dec = int(value)
    except (TypeError, ValueError):
        raise ValueError("options.topology_round_decimals must be an integer.")
    if dec == _ROUNDING_SENTINEL:
        return None
    if dec < 0 or dec > _ROUNDING_MAX:
        raise ValueError(f"options.topology_round_decimals must be between 0 and {_ROUNDING_MAX}, or {_ROUNDING_SENTINEL} to disable.")
    return dec


def _write_text_summary(
    summary_path: Path,
    dataset_dir: Path,
    work_dir: Path,
    graph_dir: Path,
    edge_dump_dir: Optional[Path],
    module_info: Dict[str, Dict[str, Optional[str]]],
    interface_result: Dict[str, object],
    topology_result: Dict[str, object],
    node_result: Dict[str, object],
    edge_result: Dict[str, object],
    summary: Optional[Dict[str, object]] = None,
) -> None:
    lines: List[str] = []
    lines.append("=== Graph Builder Summary ===")
    lines.append(f"Dataset directory: {dataset_dir}")
    lines.append(f"Work directory: {work_dir}")
    lines.append(f"Graph directory: {graph_dir}")
    if edge_dump_dir is not None:
        lines.append(f"Edge feature directory: {edge_dump_dir}")
    else:
        lines.append("Edge feature directory: (not written)")

    lines.append("")
    lines.append("Modules:")
    lines.append(
        f"  Interface: {_format_module(module_info['interface']['id'], module_info['interface']['alias'])}"
        f" (jobs={module_info['interface']['jobs']})"
    )
    lines.append(
        f"  Topology : {_format_module(module_info['topology']['id'], module_info['topology']['alias'])}"
        f" (jobs={module_info['topology']['jobs']})"
    )
    lines.append(
        f"  Node     : {_format_module(module_info['node']['id'], module_info['node']['alias'])}"
        f" (jobs={module_info['node']['jobs']})"
    )
    lines.append(
        f"  Edge     : {_format_module(module_info['edge']['id'], module_info['edge']['alias'])}"
        f" (jobs={module_info['edge']['jobs']})"
    )

    def _stage_header(title: str) -> None:
        lines.append("")
        lines.append(f"[{title}]")

    # Interface stage
    _stage_header("Interface Features")
    lines.append(f"  Successes : {interface_result['success']}")
    lines.append(f"  Failures  : {len(interface_result['failures'])}")
    lines.append(f"  Logs dir  : {interface_result['log_dir']}")
    if interface_result["failures"]:
        lines.append("  Failure details:")
        for pdb_path, log_path, error in interface_result["failures"]:
            lines.append(
                "    - {}: {} (log: {})".format(
                    _relative_path(Path(pdb_path), dataset_dir),
                    error,
                    log_path,
                )
            )

    # Topology stage
    _stage_header("Topology Features")
    lines.append(f"  Successes : {topology_result['success']}")
    lines.append(f"  Failures  : {len(topology_result['failures'])}")
    lines.append(f"  Logs dir  : {topology_result['log_dir']}")
    if topology_result["failures"]:
        lines.append("  Failure details:")
        for pdb_path, log_path, error in topology_result["failures"]:
            lines.append(
                "    - {}: {} (log: {})".format(
                    _relative_path(Path(pdb_path), dataset_dir),
                    error,
                    log_path,
                )
            )

    # Node stage
    _stage_header("Node Features")
    lines.append(f"  Successes : {node_result['success']}")
    lines.append(f"  Failures  : {len(node_result['failures'])}")
    lines.append(f"  Logs dir  : {node_result['log_dir']}")
    if node_result["failures"]:
        lines.append("  Failure details:")
        for model_key, log_path, error in node_result["failures"]:
            lines.append(f"    - {model_key}: {error} (log: {log_path})")

    # Edge stage
    _stage_header("Graph (.pt) Files")
    lines.append(f"  Processed : {edge_result['processed']}")
    lines.append(f"  Successes : {edge_result['success']}")
    lines.append(f"  Failures  : {len(edge_result['failures'])}")
    lines.append(f"  Logs dir  : {edge_result['log_dir']}")
    lines.append(f"  Run log   : {edge_result['run_log']}")
    if edge_result["failures"]:
        lines.append("  Failure details:")
        for model_key, error, log_path in edge_result["failures"]:
            lines.append(f"    - {model_key}: {error} (log: {log_path})")

    if summary:
        dims = summary.get("feature_dims") if isinstance(summary, dict) else None
        if isinstance(dims, dict):
            _stage_header("Feature Dimensions")
            for key in ("topology_feature_dim", "node_feature_dim", "edge_feature_dim"):
                if key in dims:
                    lines.append(f"  {key}: {dims.get(key)}")
        meta_map = summary.get("metadata_map") if isinstance(summary, dict) else None
        if isinstance(meta_map, dict):
            _stage_header("Metadata Map")
            for name, info in meta_map.items():
                description = ""
                path_value = ""
                if isinstance(info, dict):
                    description = info.get("description", "")
                    path_value = info.get("path", "")
                lines.append(f"  {name}: {description}")
                if path_value:
                    lines.append(f"    path: {path_value}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_graph_validation(
    graph_dir: Path,
    *,
    manifest_path: Optional[Path] = None,
    workers: int = 0,
    progress_interval: float = 15.0,
    ignore_metadata: bool = False,
) -> Dict[str, object]:
    if validate_graphs is None:
        raise RuntimeError("new_validate_graphs is unavailable; cannot run graph validation.")
    manifest = manifest_path or (graph_dir / "graph_manifest.json")
    exit_code = validate_graphs(
        graph_dir,
        manifest,
        True,
        None,
        ignore_metadata,
        workers=workers,
        progress_interval=progress_interval,
    )
    return {"manifest": str(manifest), "exit_code": exit_code}


def _maybe_run_graph_validation(
    *,
    enabled: bool,
    graph_dir: Path,
    workers: int,
    progress_interval: float,
    ignore_metadata: bool = False,
) -> Dict[str, object]:
    if not enabled:
        return {"enabled": False, "exit_code": None}
    payload = _run_graph_validation(
        graph_dir,
        workers=workers,
        progress_interval=progress_interval,
        ignore_metadata=ignore_metadata,
    )
    payload["enabled"] = True
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph_builder2.py",
        description="Generate PyG graph files using configurable feature modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Input directory containing PDB structures.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working directory for intermediate files.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Destination directory for .pt graph files.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store run logs.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Maximum worker count for parallel stages. Overrides config defaults when provided.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=None,
        help="Optional path to features.yaml; defaults to <work-dir>/features.yaml if available.",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List available feature modules and exit.",
    )
    parser.add_argument(
        "--list-modules-format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Formatting to use when listing modules (text, markdown, or json).",
    )
    parser.add_argument(
        "--create-feature-config",
        action="store_true",
        help="Generate a feature-config.yaml enumerating all modules and exit.",
    )
    parser.add_argument(
        "--include-alternates",
        action="store_true",
        help="When used with --create-feature-config, append fully-detailed commented blocks for alternate modules.",
    )
    edge_dump_group = parser.add_mutually_exclusive_group()
    edge_dump_group.add_argument(
        "--dump-edges",
        dest="dump_edges",
        action="store_const",
        const=True,
        default=None,
        help="Force per-structure edge CSV dumps on (default behaviour).",
    )
    edge_dump_group.add_argument(
        "--no-dump-edges",
        dest="dump_edges",
        action="store_const",
        const=False,
        help="Disable writing per-structure edge CSV dumps.",
    )
    parser.add_argument(
        "--no-sort-artifacts",
        action="store_true",
        help=(
            "Disable belt-and-suspenders sorting for topology/node/edge CSV artifacts. "
            "Interface ordering and in-graph edge ordering remain deterministic."
        ),
    )
    parser.add_argument(
        "--pdb-warnings",
        action="store_true",
        help="Emit Bio.PDB structure parsing warnings instead of suppressing them.",
    )
    parser.add_argument(
        "--no-validate-graphs",
        dest="validate_graphs",
        action="store_false",
        help="Disable post-run graph validation/manifest writing (enabled by default).",
    )
    parser.add_argument(
        "--validate-graphs-workers",
        type=int,
        default=0,
        help="Worker processes for graph validation (0 uses all available CPUs).",
    )
    parser.add_argument(
        "--validate-graphs-progress-interval",
        type=float,
        default=15.0,
        help="Seconds between validation progress logs (0 disables periodic logs).",
    )
    parser.set_defaults(validate_graphs=True)
    return parser


def parse_args(argv: Optional[List[str]] = None) -> Optional[argparse.Namespace]:
    parser = _build_parser()
    arg_list = list(argv) if argv is not None else sys.argv[1:]

    if not arg_list or any(flag in arg_list for flag in ("-h", "--help")):
        parser.print_help()
        return None

    args = parser.parse_args(arg_list)

    if args.list_modules or args.create_feature_config:
        return args

    missing = []
    if args.dataset_dir is None:
        missing.append("--dataset-dir")
    if args.work_dir is None:
        missing.append("--work-dir")
    if args.graph_dir is None:
        missing.append("--graph-dir")
    if args.log_dir is None:
        missing.append("--log-dir")
    if args.feature_config is None:
        missing.append("--feature-config")
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    return args


def _setup_logging(run_info: LogDirectoryInfo) -> None:
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(run_info.run_dir / "graph_builder.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOG.addHandler(console_handler)


def _resolve_feature_config(args: argparse.Namespace) -> FeatureSelection:
    user_path = Path(args.feature_config)
    if not user_path.exists():
        LOG.error("Feature configuration not found at %s", user_path)
        raise FileNotFoundError(f"Requested feature-config not found: {user_path}")
    LOG.info("Loading feature configuration from %s", user_path)
    return load_feature_config(user_path)


def _instantiate_modules(selection: FeatureSelection) -> Dict[str, object]:
    try:
        from .modules import instantiate_module as _instantiate_module  # type: ignore
    except ImportError:  # pragma: no cover - fallback for direct execution
        from modules import instantiate_module as _instantiate_module  # type: ignore

    modules = {
        "interface": _instantiate_module(selection.interface["module"], **selection.interface.get("params", {})),
        "topology": _instantiate_module(selection.topology["module"], **selection.topology.get("params", {})),
        "node": _instantiate_module(selection.node["module"], **selection.node.get("params", {})),
        "edge": _instantiate_module(selection.edge["module"], **selection.edge.get("params", {})),
    }
    return modules


def _validate_feature_selection(selection: FeatureSelection) -> None:
    try:
        from .modules import get_module_class as _get_module_class  # type: ignore
    except ImportError:  # pragma: no cover - fallback for direct execution
        from modules import get_module_class as _get_module_class  # type: ignore

    stage_map = {
        "interface": selection.interface,
        "topology": selection.topology,
        "node": selection.node,
        "edge": selection.edge,
    }
    for stage, config in stage_map.items():
        module_id = config.get("module")
        if not module_id:
            raise ValueError(f"Stage '{stage}' is missing a module identifier.")
        module_cls = _get_module_class(module_id)
        module_cls.validate_params(config.get("params", {}))

    for stage, config in selection.extras.items():
        module_id = config.get("module")
        if not module_id:
            raise ValueError(f"Stage '{stage}' is missing a module identifier.")
        module_cls = _get_module_class(module_id)
        module_cls.validate_params(config.get("params", {}))


def _apply_job_defaults(modules: Dict[str, object], jobs: Optional[int]) -> None:
    for key in ("interface", "topology", "node", "edge"):
        module = modules.get(key)
        if module and isinstance(module.params, dict) and "jobs" in module.params:
            if module.params["jobs"] in (None, "auto"):
                module.params["jobs"] = jobs


def _determine_effective_jobs(
    cli_jobs: Optional[int],
    config_jobs: Optional[int],
) -> Tuple[Optional[int], str]:
    if cli_jobs is not None:
        return max(1, int(cli_jobs)), "cli"
    if isinstance(config_jobs, int) and config_jobs > 0:
        return config_jobs, "config"
    return None, "module"


def _collect_module_templates() -> Dict[str, List[Dict[str, object]]]:
    """Return config templates + summaries for every registered module."""
    try:
        from .modules import list_modules as _list_modules, get_module_class as _get_module_class  # type: ignore
    except ImportError:  # pragma: no cover
        from modules import list_modules as _list_modules, get_module_class as _get_module_class  # type: ignore

    templates: Dict[str, List[Dict[str, object]]] = {}
    for meta in _list_modules():
        module_cls = _get_module_class(meta.module_id)
        snippet: Dict[str, object] = {}
        if hasattr(module_cls, "config_template"):
            snippet = dict(module_cls.config_template())  # type: ignore[attr-defined]
        else:  # pragma: no cover - backward compatibility helper
            snippet = {"module": meta.module_id, "params": dict(meta.defaults)}
        params = dict(snippet.get("params", {}))  # make a copy to keep class defaults immutable
        alias_value = snippet.get("alias") or getattr(module_cls, "default_alias", None)
        dim_hint = _extract_dim_hint(alias_value, params)
        entry = {
            "module_id": snippet.get("module", meta.module_id),
            "alias": alias_value,
            "params": params,
            "param_comments": dict(snippet.get("param_comments", {})),
            "summary": meta.summary or "",
            "description": meta.description or "",
            "dim_hint": dim_hint,
        }
        templates.setdefault(meta.module_kind, []).append(entry)
        alternates = snippet.get("alternates")
        if isinstance(alternates, list):
            for alt in alternates:
                alt_params = dict(alt.get("params", params))
                alt_alias = alt.get("alias") or getattr(module_cls, "default_alias", None)
                alt_dim_hint = _extract_dim_hint(alt_alias, alt_params)
                templates.setdefault(meta.module_kind, []).append(
                    {
                        "module_id": alt.get("module", meta.module_id),
                        "alias": alt_alias,
                        "params": alt_params,
                        "param_comments": dict(alt.get("param_comments", {})),
                        "summary": alt.get("summary", meta.summary or ""),
                        "description": alt.get("description", meta.description or ""),
                        "dim_hint": alt_dim_hint,
                    }
                )

    for entries in templates.values():
        entries.sort(key=lambda item: item["module_id"])  # type: ignore[index]
    return templates


def _select_default_template(kind: str, entries: List[Dict[str, object]]) -> Dict[str, object]:
    preferred = _TEMPLATE_DEFAULTS.get(kind)
    if preferred:
        for entry in entries:
            if entry["module_id"] == preferred:
                return entry
    return entries[0]


def _format_params_block(
    params: Dict[str, object],
    comments: Optional[Dict[str, str]] = None,
    indent: int = 2,
) -> List[str]:
    pad = " " * indent
    if not params:
        return [f"{pad}params: {{}}"]
    lines = [f"{pad}params:"]
    for key, value in params.items():
        lines.extend(_format_param_entry(key, value, comments or {}, indent + 2))
    return lines


def _format_param_entry(
    key: str,
    value: object,
    comments: Mapping[str, str],
    indent: int,
) -> List[str]:
    indent_str = " " * indent
    comment = f"  # {comments[key]}" if key in comments else ""

    if key == "element_filters" and _is_list_of_sequences(value):
        lines = [f"{indent_str}{key}:"]
        for entry in value:  # type: ignore[assignment]
            entry_dump = _format_flow_sequence(entry)  # type: ignore[arg-type]
            lines.append(f"{indent_str}  - {entry_dump}")
        return lines

    if isinstance(value, (list, tuple)) and _is_scalar_sequence(value):
        seq_dump = _format_flow_sequence(value)
        return [f"{indent_str}{key}: {seq_dump}{comment}"]

    if isinstance(value, Mapping):
        lines = [f"{indent_str}{key}:"]
        dumped = yaml.safe_dump(value, sort_keys=False).strip()
        if dumped:
            for line in dumped.splitlines():
                lines.append(" " * (indent + 2) + line)
        return lines

    if isinstance(value, (list, tuple)):
        lines = [f"{indent_str}{key}:"]
        dumped = yaml.safe_dump(value, sort_keys=False).strip()
        if dumped:
            for line in dumped.splitlines():
                lines.append(" " * (indent + 2) + line)
        return lines

    scalar = _dump_scalar(value)
    return [f"{indent_str}{key}: {scalar}{comment}"]


def _is_list_of_sequences(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(entry, (list, tuple)) for entry in value)


def _is_scalar_sequence(value: Sequence[object]) -> bool:
    return all(not isinstance(entry, (list, tuple, Mapping)) for entry in value)


def _format_flow_sequence(value: Sequence[object]) -> str:
    dumped = yaml.safe_dump(list(value), default_flow_style=True)
    return dumped.splitlines()[0] if dumped else "[]"


def _dump_scalar(value: object) -> str:
    dumped = yaml.safe_dump(value, default_flow_style=False)
    return dumped.splitlines()[0] if dumped else "null"


def _format_string_line(
    key: str,
    value: object,
    indent: int = 2,
    allow_empty: bool = False,
) -> Optional[str]:
    if value is None or value == "":
        if not allow_empty:
            return None
        encoded = json.dumps("")
    else:
        encoded = json.dumps(str(value))
    return f"{' ' * indent}{key}: {encoded}"


def _render_stage_block(
    stage: str,
    entry: Dict[str, object],
    *,
    comment_prefix: str = "",
) -> List[str]:
    lines: List[str] = []

    def wrap(text: str) -> str:
        return f"{comment_prefix}{text}" if comment_prefix else text

    alias_hint = entry.get("alias")
    module_value = f"  module: {entry.get('module_id')}"
    if alias_hint:
        module_value += f"  # alias: {alias_hint}"

    lines.append(wrap(f"{stage}:"))
    lines.append(wrap(module_value))

    dim_hint = entry.get("dim_hint")
    if dim_hint is None:
        dim_hint = _extract_dim_hint(entry.get("alias"), entry.get("params"))
    for meta_key in ("alias", "summary", "description"):
        meta_line = _format_string_line(meta_key, entry.get(meta_key), allow_empty=True)
        if meta_line:
            lines.append(wrap(meta_line))
    if dim_hint is not None:
        lines.append(wrap(f"  # dim: {dim_hint}"))

    param_lines = _format_params_block(
        entry.get("params", {}),
        comments=entry.get("param_comments"),
        indent=2,
    )
    for param_line in param_lines:
        lines.append(wrap(param_line))

    return lines


def _ordered_kinds(module_kinds: Iterable[str]) -> List[str]:
    ordered = list(_STAGE_ORDER)
    for kind in sorted(module_kinds):
        if kind not in ordered:
            ordered.append(kind)
    return ordered


def _extract_dim_hint(alias: Optional[str], params: Optional[Dict[str, object]] = None) -> Optional[int]:
    """Best-effort dimension hint from alias text, respecting variant when present."""
    if not alias:
        return None
    alias_lower = alias.lower()
    if "dynamic" in alias_lower:
        return None

    variant = None
    if isinstance(params, dict):
        v = params.get("variant")
        if isinstance(v, str):
            variant = v.strip().lower()

    # If the alias encodes both lean/heavy separated by '|', pick the segment matching variant.
    segments = [seg.strip() for seg in alias.split("|") if seg.strip()]
    segment = alias
    if len(segments) > 1 and variant:
        if variant == "lean":
            segment = segments[0]
        elif variant == "heavy":
            segment = segments[-1]
    elif len(segments) > 1:
        # default to first segment for multi-part aliases when variant unknown
        segment = segments[0]

    matches = re.findall(r"(\d+)\s*D|Edge\s+(\d+)", segment, flags=re.IGNORECASE)
    if not matches:
        return None
    numbers: List[int] = []
    for a, b in matches:
        if a:
            numbers.append(int(a))
        if b:
            numbers.append(int(b))
    return numbers[-1] if numbers else None


def _render_feature_config_template(
    modules_by_kind: Dict[str, List[Dict[str, object]]],
    *,
    include_alternates: bool = False,
) -> str:
    lines: List[str] = []
    lines.append("# Auto-generated feature-config template")
    lines.append("# Pick one module per required stage. Edit parameter values as needed.")
    lines.append("# Run `./run_graph_builder2.sh --list-modules` for detailed descriptions.")
    lines.append("defaults:")
    lines.append("  jobs: 16  # Optional global worker override; remove if unused.")
    lines.append("")
    lines.append("options:")
    lines.append(
        "  # topology_round_decimals: 12  # Round topology numeric columns to N decimals (0-15); -1 or omit to disable (recommended 12 when enabling)."
    )

    kind_order = _ordered_kinds(modules_by_kind.keys())

    for stage in kind_order:
        lines.append("")
        stage_header = f"# {'REQUIRED' if stage in _REQUIRED_STAGES else 'OPTIONAL'} stage: {stage}"
        lines.append(stage_header)
        options = modules_by_kind.get(stage, [])
        if not options:
            lines.append(f"#   No registered {stage} modules were found.")
            continue

        default_entry = _select_default_template(stage, options)
        lines.extend(_render_stage_block(stage, default_entry))

        alternates = [entry for entry in options if entry is not default_entry]
        if alternates:
            lines.append("")
            if include_alternates:
                lines.append(f"# Alternate {stage} modules (uncomment to use):")
                for alt in alternates:
                    lines.extend(_render_stage_block(stage, alt, comment_prefix="# "))
                    lines.append("#")
            else:
                lines.append(f"# Alternate {stage} modules:")
                for alt in alternates:
                    alias_note = f" (alias: {alt.get('alias')})" if alt.get("alias") else ""
                    summary = alt.get("summary") or "No summary provided."
                    lines.append(f"#   - {alt['module_id']}{alias_note}: {summary}")

    lines.append("")
    lines.append("# OPTIONAL custom stages ------------------------------------------------")
    lines.append("# Add new top-level keys (e.g., `mol`) only if the builder has logic to process them.")
    lines.append("# Each custom block must still follow the `module` + `params` structure shown above.")
    lines.append("# Example:")
    lines.append("# mol:")
    lines.append("#   module: custom/mol_stage/v1")
    lines.append("#   params: {}")
    return "\n".join(lines).strip() + "\n"


def _iter_registered_modules(kind: Optional[str] = None) -> Iterable[Tuple["FeatureModuleMetadata", Type]]:
    try:
        from .modules import list_modules as _list_modules, get_module_class as _get_module_class  # type: ignore
    except ImportError:  # pragma: no cover
        from modules import list_modules as _list_modules, get_module_class as _get_module_class  # type: ignore

    metas = sorted(_list_modules(kind=kind), key=lambda m: m.module_id)
    for meta in metas:
        module_cls = _get_module_class(meta.module_id)
        yield meta, module_cls


def _format_module_listing(meta: "FeatureModuleMetadata", module_cls: Type) -> List[str]:
    alias = getattr(module_cls, "default_alias", None)
    alias_note = f" (alias: {alias})" if alias else ""
    summary = meta.summary or meta.description or "No summary provided."
    dim_hint = _extract_dim_hint(alias)

    lines = [
        f"  {meta.module_id}{alias_note}",
        f"    summary : {summary}",
    ]
    if dim_hint is not None:
        lines.append(f"    dim     : {dim_hint}")

    try:
        param_desc = module_cls.list_params()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - legacy modules
        param_desc = dict(meta.parameters)

    if param_desc:
        lines.append("    params  :")
        for name, desc in param_desc.items():
            default_value = meta.defaults.get(name, "∅")
            lines.append(f"      - {name} (default={default_value}): {desc}")
    return lines


def _module_to_dict(meta: "FeatureModuleMetadata", module_cls: Type) -> Dict[str, object]:
    try:
        param_desc = module_cls.list_params()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - legacy modules
        param_desc = dict(meta.parameters)
    defaults = dict(meta.defaults)
    return {
        "id": meta.module_id,
        "kind": meta.module_kind,
        "alias": getattr(module_cls, "default_alias", None),
        "summary": meta.summary or meta.description,
        "parameters": param_desc,
        "defaults": defaults,
    }


def _list_registered_modules(output_format: str = "text") -> None:
    module_map: Dict[str, List[Tuple["FeatureModuleMetadata", Type]]] = {}
    for meta, module_cls in _iter_registered_modules():
        module_map.setdefault(meta.module_kind, []).append((meta, module_cls))

    kinds = _ordered_kinds(module_map.keys())

    if output_format == "json":
        payload: Dict[str, List[Dict[str, object]]] = {}
        for kind in kinds:
            entries = module_map.get(kind, [])
            payload[kind] = [_module_to_dict(meta, cls) for meta, cls in sorted(entries, key=lambda item: item[0].module_id)]
        print(json.dumps(payload, indent=2))
        return

    if output_format == "markdown":
        lines: List[str] = []
        lines.append("# Registered Feature Modules")
        lines.append("")
        required = ", ".join(sorted(_REQUIRED_STAGES))
        lines.append(f"_Required stages:_ `{required}`")
        lines.append("")
        for kind in kinds:
            header = f"### {kind} modules"
            if kind in _REQUIRED_STAGES:
                header += " (required)"
            else:
                header += " (optional)"
            lines.append(header)
            entries = module_map.get(kind, [])
            if not entries:
                lines.append("_No registered modules._")
                lines.append("")
                continue
            for meta, module_cls in sorted(entries, key=lambda item: item[0].module_id):
                alias = getattr(module_cls, "default_alias", None)
                dim_hint = _extract_dim_hint(alias)
                dim_note = f" [dim={dim_hint}]" if dim_hint is not None else ""
                alias_note = f" *(alias: {alias})*" if alias else ""
                summary = meta.summary or meta.description or "No summary provided."
                lines.append(f"- `{meta.module_id}`{alias_note}{dim_note} — {summary}")
                try:
                    param_desc = module_cls.list_params()  # type: ignore[attr-defined]
                except AttributeError:
                    param_desc = dict(meta.parameters)
                defaults = meta.defaults
                if param_desc:
                    for name, desc in param_desc.items():
                        default_value = defaults.get(name, "∅")
                        lines.append(f"  - **{name}** (default=`{default_value}`): {desc}")
            lines.append("")
        print("\n".join(lines).rstrip() + "\n")
        return

    print("Select exactly one module per category (interface → topology → node → edge) when configuring features.\n")
    print("Need to blend or extend feature logic? Copy an existing module (interface/topology/node/edge), "
          "adapt or compose it, add @register_feature_module, and reference the new module ID in feature-config.yaml. "
          "Each run still selects exactly one interface, one topology, one node, and one edge module.\n")

    for kind in kinds:
        print(f"{kind.upper()} modules")
        entries = module_map.get(kind, [])
        if not entries:
            print("  (no registered modules)\n")
            continue
        printed_once = False
        for meta, module_cls in sorted(entries, key=lambda item: item[0].module_id):
            if printed_once:
                print()
            printed_once = True
            for line in _format_module_listing(meta, module_cls):
                print(line)
        print()

    diagram = """
Input: PDB structures
        │
        ▼
+------------------------------------------------------------------------+
| Interface module                                                       |
|   e.g.                                                                 |
|     - interface/polar_cutoff/v1 (per residue: 3D coordinates, default) |
+------------------------------------------------------------------------+
        │  produces residue IDs + coords
        ▼
+------------------------------------------------------------------------+
| Topology module                                                        |
|   e.g.                                                                 |
|     - topology/persistence_basic/v1 (140-D default)                    |
+------------------------------------------------------------------------+
        │  emits per-residue topology CSV
        │
        ├─► (along with interface.csv + PDB) feeds node stage
        │
        ▼
+------------------------------------------------------------------------+
| Node module                                                            |
|   e.g.                                                                 |
|     - node/dssp_topo_merge/v1 (default)                                |
+------------------------------------------------------------------------+
        │  outputs ordered node_features.csv
        ▼
+------------------------------------------------------------------------+
| Edge module                                                            |
|   e.g.                                                                 |
|     - edge/legacy_band/v11  (11-D default)                             |
|     - edge/multi_scale/v24 (24-D multi-scale)                          |
+------------------------------------------------------------------------+
        │  combines interface + node + PDB
        ▼
Final output: torch_geometric Data (.pt)
"""
    print(diagram.strip())


def write_feature_config(output_path: Path, include_alternates: bool = False) -> None:
    templates = _collect_module_templates()
    template_text = _render_feature_config_template(templates, include_alternates=include_alternates)
    output_path.write_text(template_text, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args is None:
        return 0
    if args.list_modules:
        _list_registered_modules(output_format=getattr(args, "list_modules_format", "text"))
        return 0
    if getattr(args, "create_feature_config", False):
        work_dir = Path(args.work_dir or ".").resolve() if args.work_dir else Path.cwd()
        output_path = work_dir / "example.feature-config.yaml"
        write_feature_config(output_path, include_alternates=getattr(args, "include_alternates", False))
        print("Feature configuration template written to ./example.feature-config.yaml")
        return 0

    dataset_dir = Path(args.dataset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    graph_dir = Path(args.graph_dir).resolve()
    log_root = Path(args.log_dir).resolve()
    feature_config_path = Path(args.feature_config).expanduser().resolve()
    args.feature_config = str(feature_config_path)
    cli_jobs = args.jobs if args.jobs is not None else None
    sort_artifacts = not bool(getattr(args, "no_sort_artifacts", False))

    try:
        ensure_tree_readable(dataset_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        ensure_tree_readwrite(work_dir)
        ensure_tree_readwrite(graph_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        ensure_tree_readwrite(log_root)
    except PermissionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    run_info = prepare_log_directory(log_root, run_prefix="graph_builder")
    _setup_logging(run_info)
    configure_pdb_parser(bool(getattr(args, "pdb_warnings", False)))

    round_decimals: Optional[int] = None
    try:
        selection = _resolve_feature_config(args)
        _validate_feature_selection(selection)
        round_decimals = _resolve_round_decimals(selection.options.get("topology_round_decimals"))
        if selection.options.get("topology_round_decimals") is not None:
            selection.options["topology_round_decimals"] = (
                _ROUNDING_SENTINEL if round_decimals is None else round_decimals
            )
    except Exception as exc:
        message = f"Feature configuration invalid: {exc}"
        LOG.error(message)
        print(f"Error: {message}", file=sys.stderr)
        return 2

    LOG.info("Prepared log directory: %s", run_info.run_dir)
    LOG.info("Dataset directory: %s", dataset_dir)
    LOG.info("Work directory: %s", work_dir)
    LOG.info("Graph directory: %s", graph_dir)

    default_jobs_override = selection.options.get("default_jobs")
    effective_jobs, source = _determine_effective_jobs(cli_jobs, default_jobs_override)
    if source == "cli":
        LOG.info("Jobs: %s (from CLI)", effective_jobs)
    elif source == "config":
        LOG.info("Jobs: %s (override from feature config)", effective_jobs)
    else:
        LOG.info("Jobs: auto (fall back to per-module defaults)")
    modules = _instantiate_modules(selection)
    _apply_job_defaults(modules, effective_jobs)

    interface_module: InterfaceFeatureModule = modules["interface"]  # type: ignore[assignment]
    topology_module: TopologyFeatureModule = modules["topology"]  # type: ignore[assignment]
    node_module: NodeFeatureModule = modules["node"]  # type: ignore[assignment]
    edge_module: EdgeFeatureModule = modules["edge"]  # type: ignore[assignment]

    interface_alias = selection.interface.get("alias")
    topology_alias = selection.topology.get("alias")
    node_alias = selection.node.get("alias")
    edge_alias = selection.edge.get("alias")

    interface_jobs = _resolve_jobs(interface_module, effective_jobs)
    topology_jobs = _resolve_jobs(topology_module, effective_jobs)
    node_jobs = _resolve_jobs(node_module, effective_jobs)
    edge_jobs = _resolve_jobs(edge_module, effective_jobs)

    LOG.info(
        "Interface module: %s (jobs=%s)",
        _format_module(interface_module.metadata().module_id, interface_alias),
        interface_jobs,
    )
    LOG.info(
        "Topology module : %s (jobs=%s)",
        _format_module(topology_module.metadata().module_id, topology_alias),
        topology_jobs,
    )
    LOG.info(
        "Node module     : %s (jobs=%s)",
        _format_module(node_module.metadata().module_id, node_alias),
        node_jobs,
    )
    LOG.info(
        "Edge module     : %s (jobs=%s)",
        _format_module(edge_module.metadata().module_id, edge_alias),
        edge_jobs,
    )
    LOG.info(
        "Artifact CSV sorting: %s (interface ordering always deterministic)",
        "enabled" if sort_artifacts else "disabled via --no-sort-artifacts",
    )
    if round_decimals is None:
        LOG.info("Topology rounding: disabled (set options.topology_round_decimals to enable)")
    else:
        LOG.info("Topology rounding: enabled (decimals=%d)", round_decimals)

    pdb_files = sorted(dataset_dir.rglob("*.pdb"))
    if not pdb_files:
        LOG.error("No PDB files found under %s", dataset_dir)
        return 1
    LOG.info("Identified %d PDB files for processing.", len(pdb_files))

    interface_desc = interface_module.describe()
    interface_desc["alias"] = interface_alias
    interface_desc["jobs"] = interface_jobs
    topology_desc = topology_module.describe()
    topology_desc["alias"] = topology_alias
    topology_desc["jobs"] = topology_jobs
    node_desc = node_module.describe()
    node_desc["alias"] = node_alias
    node_desc["jobs"] = node_jobs
    edge_desc = edge_module.describe()
    edge_desc["alias"] = edge_alias
    edge_desc["jobs"] = edge_jobs

    summary: Dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "work_dir": str(work_dir),
        "graph_dir": str(graph_dir),
        "log_dir": str(run_info.run_dir),
        "modules": {
            "interface": interface_desc,
            "topology": topology_desc,
            "node": node_desc,
            "edge": edge_desc,
        },
    }

    options = selection.options
    edge_dump_enabled = _resolve_edge_dump(getattr(args, "dump_edges", None), options.get("edge_dump"))
    options["edge_dump"] = edge_dump_enabled
    edge_dump_cli_dir = getattr(args, "edge_dump_dir", None)
    edge_dump_config_dir = options.get("edge_dump_dir")
    resolved_dump_dir = _resolve_edge_dump_dir(work_dir, edge_dump_cli_dir, edge_dump_config_dir)

    LOG.info("Beginning interface stage (jobs=%s)", interface_jobs)
    interface_result = interface_module.extract_interfaces(
        pdb_paths=pdb_files,
        dataset_dir=dataset_dir,
        work_dir=work_dir,
        log_dir=run_info.run_dir,
    )
    summary["interface"] = interface_result
    LOG.info("Interface stage complete: %d success, %d failures", interface_result["success"], len(interface_result["failures"]))
    LOG.info("Interface feature extraction elapsed time: %.2f s", interface_result.get("elapsed", 0.0))

    LOG.info("Beginning topology stage (jobs=%s)", topology_jobs)
    if round_decimals is not None:
        LOG.info("Topology rounding: enabled (decimals=%d)", round_decimals)
    topology_result = topology_module.generate_topology(
        pdb_paths=pdb_files,
        dataset_dir=dataset_dir,
        interface_dir=interface_result["output_dir"],
        work_dir=work_dir,
        log_dir=run_info.run_dir,
        sort_artifacts=sort_artifacts,
        round_decimals=round_decimals,
    )
    summary["topology"] = topology_result
    LOG.info("Topology stage complete: %d success, %d failures", topology_result["success"], len(topology_result["failures"]))
    LOG.info("Topology feature extraction elapsed time: %.2f s", topology_result.get("elapsed", 0.0))

    structure_map = index_structures(dataset_dir, (".pdb",))
    LOG.info("Beginning node stage (jobs=%s)", node_jobs)
    node_result = node_module.generate_nodes(
        dataset_dir=dataset_dir,
        structure_map=structure_map,
        interface_dir=interface_result["output_dir"],
        topology_dir=topology_result["output_dir"],
        work_dir=work_dir,
        log_dir=run_info.run_dir,
        sort_artifacts=sort_artifacts,
    )
    summary["node"] = node_result
    LOG.info("Node stage complete: %d success, %d failures", node_result["success"], len(node_result["failures"]))
    LOG.info("Node feature extraction elapsed time: %.2f s", node_result.get("elapsed", 0.0))

    edge_dump_dir: Optional[Path] = resolved_dump_dir if edge_dump_enabled else None
    edge_dump_log_start = time.perf_counter() if edge_dump_enabled else None

    summary["edge_dump"] = {
        "enabled": edge_dump_enabled,
        "directory": str(edge_dump_dir) if edge_dump_dir is not None else None,
        "configured_directory": str(resolved_dump_dir) if resolved_dump_dir is not None else None,
    }
    builder_info = build_builder_info(
        feature_config_path=feature_config_path,
        edge_dump_enabled=edge_dump_enabled,
        edge_dump_dir=edge_dump_dir,
        configured_edge_dump_dir=resolved_dump_dir,
        selection_options=selection.options,
    )
    summary["builder"] = builder_info

    if edge_dump_enabled:
        LOG.info("Edge dumps enabled; writing CSVs to:")
        LOG.info("  %s", edge_dump_dir)
    else:
        if edge_dump_cli_dir is not None or (
            isinstance(edge_dump_config_dir, str) and edge_dump_config_dir.strip()
        ):
            LOG.info("Edge dumps disabled; configured dump directory %s will be ignored.", resolved_dump_dir)
        else:
            LOG.info("Edge dumps disabled (use --dump-edges to re-enable).")
    if edge_dump_log_start is not None:
        LOG.info("Edge dump preparation elapsed time: %.2f s", time.perf_counter() - edge_dump_log_start)

    LOG.info("Beginning graph assembly (.pt output) stage (jobs=%s)", edge_jobs)
    try:
        from .lib.edge_runner import run_edge_stage as _run_edge_stage  # type: ignore
    except ImportError:  # pragma: no cover - fallback for direct execution
        from lib.edge_runner import run_edge_stage as _run_edge_stage  # type: ignore

    graph_stage_start = time.perf_counter()

    edge_result = _run_edge_stage(
        dataset_dir=dataset_dir,
        interface_dir=interface_result["output_dir"],
        topology_dir=topology_result["output_dir"],
        node_dir=node_result["output_dir"],
        output_dir=graph_dir,
        log_dir=run_info.run_dir / "edge_logs",
        edge_module=edge_module,
        jobs=edge_jobs,
        edge_dump_dir=edge_dump_dir,
        builder_info=builder_info,
        sort_artifacts=sort_artifacts,
        module_registry=summary.get("modules"),
    )
    summary["edge"] = edge_result
    graph_stage_elapsed = time.perf_counter() - graph_stage_start
    LOG.info("Graph assembly (.pt output) complete: %d success, %d failures", edge_result["success"], len(edge_result["failures"]))
    LOG.info(
        "Graph (.pt) generation elapsed time: %.2f s (edge stage: %.2f s)",
        edge_result.get("elapsed", 0.0),
        graph_stage_elapsed,
    )
    summary["edge"]["graph_stage_elapsed"] = graph_stage_elapsed

    validation_failed = False
    try:
        validation_summary = _maybe_run_graph_validation(
            enabled=bool(getattr(args, "validate_graphs", True)),
            graph_dir=graph_dir,
            workers=int(getattr(args, "validate_graphs_workers", 0)),
            progress_interval=float(getattr(args, "validate_graphs_progress_interval", 15.0)),
        )
        summary["validation"] = validation_summary
        if validation_summary.get("enabled") and validation_summary.get("exit_code") not in (0, None):
            validation_failed = True
            LOG.error("Graph validation failed (exit_code=%s).", validation_summary.get("exit_code"))
    except Exception as exc:  # pragma: no cover - safety for validation failures
        validation_failed = True
        summary["validation"] = {"enabled": True, "error": str(exc)}
        LOG.error("Graph validation error: %s", exc)

    feature_dims: Dict[str, object] = {}
    metadata_map: Dict[str, Dict[str, str]] = {}
    graph_metadata_path = graph_dir / "graph_metadata.json"
    if graph_metadata_path.exists():
        try:
            graph_metadata = json.loads(graph_metadata_path.read_text(encoding="utf-8"))
            for key in ("topology_feature_dim", "node_feature_dim", "edge_feature_dim"):
                if graph_metadata.get(key) is not None:
                    feature_dims[key] = graph_metadata.get(key)
        except Exception as exc:
            LOG.warning("Unable to read graph_metadata.json for feature dims: %s", exc)
    if "edge_feature_dim" not in feature_dims and edge_result.get("edge_feature_dim") is not None:
        feature_dims["edge_feature_dim"] = edge_result.get("edge_feature_dim")
    summary["feature_dims"] = feature_dims

    metadata_map["graph_metadata.json"] = {
        "description": "Top-level metadata (dims, module registry, schema hints).",
        "path": str(graph_metadata_path),
    }
    metadata_map["topology_columns.json"] = {
        "description": "Topology column names (ID + feature columns).",
        "path": str(graph_dir / "topology_columns.json"),
    }
    metadata_map["node_columns.json"] = {
        "description": "Node feature column names (if emitted).",
        "path": str(graph_dir / "node_columns.json"),
    }
    metadata_map["edge_columns.json"] = {
        "description": "Edge feature column names (if emitted).",
        "path": str(graph_dir / "edge_columns.json"),
    }
    metadata_map["graph_manifest.json"] = {
        "description": "Validation manifest from new_validate_graphs (if enabled).",
        "path": str(graph_dir / "graph_manifest.json"),
    }
    metadata_map["graph_builder_summary.json"] = {
        "description": "Run summary metrics (machine-readable).",
        "path": str(graph_dir / "graph_builder_summary.json"),
    }
    metadata_map["graph_builder_summary.log"] = {
        "description": "Run summary log (human-readable).",
        "path": str(graph_dir / "graph_builder_summary.log"),
    }
    summary["metadata_map"] = metadata_map

    summary_log = run_info.run_dir / "graph_builder_summary.json"
    summary_log.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    LOG.info("Summary written to %s", _relative_path(summary_log, run_info.root_dir))

    text_summary_path = run_info.run_dir / "graph_builder_summary.log"
    module_info = {
        "interface": {"id": interface_module.metadata().module_id, "alias": interface_alias, "jobs": interface_jobs},
        "topology": {"id": topology_module.metadata().module_id, "alias": topology_alias, "jobs": topology_jobs},
        "node": {"id": node_module.metadata().module_id, "alias": node_alias, "jobs": node_jobs},
        "edge": {"id": edge_module.metadata().module_id, "alias": edge_alias, "jobs": edge_jobs},
    }

    _write_text_summary(
        text_summary_path,
        dataset_dir,
        work_dir,
        graph_dir,
        edge_dump_dir,
        module_info,
        interface_result,
        topology_result,
        node_result,
        edge_result,
        summary,
    )
    LOG.info("Summary log written to %s", _relative_path(text_summary_path, run_info.root_dir))

    def _copy_metadata_into_graph_dir(source: Path, destination: Path, label: str) -> None:
        try:
            shutil.copy2(source, destination)
            LOG.info("Copied %s to %s", label, _relative_path(destination, run_info.root_dir))
        except OSError as exc:
            LOG.warning("Unable to copy %s to %s: %s", label, destination, exc)

    graph_summary_json = graph_dir / "graph_builder_summary.json"
    graph_summary_log = graph_dir / "graph_builder_summary.log"
    _copy_metadata_into_graph_dir(summary_log, graph_summary_json, "graph builder summary")
    _copy_metadata_into_graph_dir(text_summary_path, graph_summary_log, "graph builder text summary")

    if edge_result["success"] == 0:
        LOG.error("No graphs produced. See logs for details.")
        return 1
    if validation_failed:
        LOG.error("Graph validation failed. See logs for details.")
        return 1

    LOG.info("Dynamic graph builder completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
