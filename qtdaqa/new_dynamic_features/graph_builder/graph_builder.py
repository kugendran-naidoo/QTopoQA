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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

try:
    from .lib.features_config import FeatureSelection, load_feature_config
    from .lib.directory_permissions import ensure_tree_readable, ensure_tree_readwrite
    from .lib.log_dirs import LogDirectoryInfo, prepare_log_directory
    from .lib.stage_common import index_structures
except ImportError:  # pragma: no cover - fallback for direct execution
    from lib.features_config import FeatureSelection, load_feature_config
    from lib.directory_permissions import ensure_tree_readable, ensure_tree_readwrite
    from lib.log_dirs import LogDirectoryInfo, prepare_log_directory
    from lib.stage_common import index_structures

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
) -> None:
    lines: List[str] = []
    lines.append("=== Graph Builder Summary ===")
    lines.append(f"Dataset directory: {dataset_dir}")
    lines.append(f"Work directory: {work_dir}")
    lines.append(f"Graph directory: {graph_dir}")
    if edge_dump_dir is not None:
        lines.append(f"Edge feature directory: {edge_dump_dir}")

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

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graph_builder.py",
        description="Generate PyG graph files using configurable feature modules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Input directory containing PDB structures.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working directory for intermediate files.")
    parser.add_argument("--graph-dir", type=Path, default=None, help="Destination directory for .pt graph files.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store run logs.")
    parser.add_argument("--jobs", type=int, default=4, help="Maximum worker count for parallel stages.")
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
        "--create-feature-config",
        action="store_true",
        help="Generate a feature-config.yaml enumerating all modules and exit.",
    )
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
    topo_params = selection.topology.get("params", {})
    filters = topo_params.get("element_filters")
    if filters is not None:
        if isinstance(filters, str):
            raise ValueError(
                "topology.params.element_filters must be a YAML list (e.g., '- [C]' entries), "
                "not a single string literal. See --create-feature-config template for examples."
            )
        if not isinstance(filters, (list, tuple)):
            raise ValueError("topology.params.element_filters must be a list of element sequences.")
        normalised = []
        for item in filters:
            if not isinstance(item, (list, tuple)):
                raise ValueError("Each element_filters entry must be a list/tuple of element symbols.")
            if not item:
                raise ValueError("Element filter sequences must contain at least one symbol.")
            for symbol in item:
                if not isinstance(symbol, str) or not symbol.strip():
                    raise ValueError("Element filter symbols must be non-empty strings.")
            normalised.append(tuple(symbol.strip() for symbol in item))
        topo_params["element_filters"] = normalised

    interface_params = selection.interface.get("params", {})
    cutoff = interface_params.get("cutoff")
    if cutoff is not None:
        try:
            cutoff_value = float(cutoff)
        except (TypeError, ValueError):
            raise ValueError("interface.params.cutoff must be numeric.")
        if cutoff_value <= 0:
            raise ValueError("interface.params.cutoff must be > 0.")
        interface_params["cutoff"] = cutoff_value

    edge_params = selection.edge.get("params", {})
    histogram_bins = edge_params.get("histogram_bins")
    if histogram_bins is not None:
        if not isinstance(histogram_bins, (list, tuple)) or len(histogram_bins) < 2:
            raise ValueError("edge.params.histogram_bins must be a list of at least two numeric values.")
        try:
            numeric_bins = [float(value) for value in histogram_bins]
        except (TypeError, ValueError):
            raise ValueError("edge.params.histogram_bins must contain only numeric values.")
        if any(b <= 0 for b in numeric_bins):
            raise ValueError("edge.params.histogram_bins must contain positive values.")
        if numeric_bins != sorted(numeric_bins):
            raise ValueError("edge.params.histogram_bins must be sorted ascending.")
        edge_params["histogram_bins"] = numeric_bins

    contact_threshold = edge_params.get("contact_threshold")
    if contact_threshold is not None:
        try:
            threshold_value = float(contact_threshold)
        except (TypeError, ValueError):
            raise ValueError("edge.params.contact_threshold must be numeric.")
        if threshold_value <= 0:
            raise ValueError("edge.params.contact_threshold must be > 0.")
        edge_params["contact_threshold"] = threshold_value

    module_id = selection.edge.get("module")
    if module_id == "edge/neo/v24":
        contact_thresholds = edge_params.get("contact_thresholds")
        if contact_thresholds is not None:
            if not isinstance(contact_thresholds, (list, tuple)) or not contact_thresholds:
                raise ValueError("edge.params.contact_thresholds must be a list of positive numbers.")
            try:
                numeric_thresholds = sorted(float(value) for value in contact_thresholds)
            except (TypeError, ValueError):
                raise ValueError("edge.params.contact_thresholds must contain only numeric values.")
            if any(value <= 0 for value in numeric_thresholds):
                raise ValueError("edge.params.contact_thresholds values must be > 0.")
            edge_params["contact_thresholds"] = numeric_thresholds
        elif contact_threshold is not None:
            edge_params["contact_thresholds"] = [edge_params.pop("contact_threshold")]

        histogram_mode = edge_params.get("histogram_mode")
        if histogram_mode is not None and not isinstance(histogram_mode, str):
            raise ValueError("edge.params.histogram_mode must be a string if specified.")

        legacy_bins = edge_params.get("legacy_histogram_bins")
        if legacy_bins is not None:
            if not isinstance(legacy_bins, (list, tuple)) or len(legacy_bins) < 2:
                raise ValueError("edge.params.legacy_histogram_bins must be a list of at least two numeric values.")
            try:
                numeric_legacy = [float(value) for value in legacy_bins]
            except (TypeError, ValueError):
                raise ValueError("edge.params.legacy_histogram_bins must contain only numeric values.")
            if numeric_legacy != sorted(numeric_legacy):
                raise ValueError("edge.params.legacy_histogram_bins must be sorted ascending.")
            edge_params["legacy_histogram_bins"] = numeric_legacy


def _apply_job_defaults(modules: Dict[str, object], jobs: Optional[int]) -> None:
    for key in ("interface", "topology", "node"):
        module = modules.get(key)
        if module and isinstance(module.params, dict) and "jobs" in module.params:
            if module.params["jobs"] in (None, "auto"):
                module.params["jobs"] = jobs


def _list_registered_modules() -> None:
    try:
        from .modules import list_modules as _list_modules  # type: ignore
    except ImportError:  # pragma: no cover
        from modules import list_modules as _list_modules  # type: ignore

    print("Select exactly one module per category (interface → topology → node → edge) when configuring features.\n")
    print("Need to blend or extend feature logic? Copy an existing module (interface/topology/node/edge), "
          "adapt or compose it, add @register_feature_module, and reference the new module ID in feature-config.yaml. "
          "Each run still selects exactly one interface, one topology, one node, and one edge module.\n")
    for kind in ("interface", "topology", "node", "edge"):
        print(f"{kind.upper()} modules")
        for meta in sorted(_list_modules(kind=kind), key=lambda m: m.module_id):
            print(f"  {meta.module_id}")
            print(f"    summary : {meta.summary}")
            if meta.parameters:
                print("    params  :")
                for name, desc in meta.parameters.items():
                    default = meta.defaults.get(name, "∅")
                    print(f"      - {name} (default={default}): {desc}")
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
|     - edge/multi_scale/v24 (24-D default)                              |
|     - edge/legacy_band/v11  (11-D legacy)                              |
+------------------------------------------------------------------------+
        │  combines interface + node + PDB
        ▼
Final output: torch_geometric Data (.pt)
"""
    print(diagram.strip())


def write_feature_config(output_path: Path) -> None:
    template = """# Minimal feature-config template
# Pick exactly one module per REQUIRED stage below.
# Run `./run_graph_builder.sh --list-modules` to discover valid module IDs.
# Remove any blocks you do not use—do not paste the module catalog output here.
defaults:
  jobs: 8  # Optional global worker override; remove if unused.

# REQUIRED STAGES -------------------------------------------------------
interface:
  module: interface/polar_cutoff/v1
  params:
    cutoff: 14.0
    coordinate_decimals: 3

topology:
  module: topology/persistence_basic/v1
  params:
    neighbor_distance: 8.0
    filtration_cutoff: 8.0
    min_persistence: 0.01

node:
  module: node/dssp_topo_merge/v1
  params:
    drop_na: false

edge:
  module: edge/multi_scale/v24
  params:
    histogram_bins: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    contact_threshold: 5.0

# Alternate hybrid edge module inspired by the legacy 11-D features.
# Uncomment the block below to start from the neo/v24 defaults.
# edge:
#   module: edge/neo/v24
#   params:
#     histogram_bins: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
#     histogram_mode: density_times_contact
#     contact_thresholds: [5.0, 10.0]
#     include_inverse_distance: true
#     include_unit_vector: true

# OPTIONAL / CUSTOM STAGES ----------------------------------------------
# Rename the key (e.g., \"mol\") and provide module/params only if the builder
# has logic to consume the stage. Otherwise leave this section out entirely.
# mol:
#   module: custom/mol_stage/v1
#   params: {}
"""
    output_path.write_text(template.strip() + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args is None:
        return 0
    if args.list_modules:
        _list_registered_modules()
        return 0
    if getattr(args, "create_feature_config", False):
        work_dir = Path(args.work_dir or ".").resolve() if args.work_dir else Path.cwd()
        output_path = work_dir / "example.feature-config.yaml"
        write_feature_config(output_path)
        print("Feature configuration template written to ./example.feature-config.yaml")
        return 0

    dataset_dir = Path(args.dataset_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    graph_dir = Path(args.graph_dir).resolve()
    log_root = Path(args.log_dir).resolve()
    jobs = max(1, args.jobs)

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

    try:
        selection = _resolve_feature_config(args)
        _validate_feature_selection(selection)
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
    if isinstance(default_jobs_override, int) and default_jobs_override > 0:
        effective_jobs = default_jobs_override
        LOG.info("Jobs: %d (override from feature config)", effective_jobs)
    else:
        effective_jobs = jobs
        LOG.info("Jobs: %d", effective_jobs)
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
    topology_result = topology_module.generate_topology(
        pdb_paths=pdb_files,
        dataset_dir=dataset_dir,
        interface_dir=interface_result["output_dir"],
        work_dir=work_dir,
        log_dir=run_info.run_dir,
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
    )
    summary["node"] = node_result
    LOG.info("Node stage complete: %d success, %d failures", node_result["success"], len(node_result["failures"]))
    LOG.info("Node feature extraction elapsed time: %.2f s", node_result.get("elapsed", 0.0))

    edge_dump_dir = None
    options = selection.options
    if options.get("edge_dump", False):
        edge_dump_dir = work_dir / "edge_features"

    LOG.info("Beginning graph assembly (.pt output) stage (jobs=%s)", edge_jobs)
    try:
        from .lib.edge_runner import run_edge_stage as _run_edge_stage  # type: ignore
    except ImportError:  # pragma: no cover - fallback for direct execution
        from lib.edge_runner import run_edge_stage as _run_edge_stage  # type: ignore

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
    )
    summary["edge"] = edge_result
    LOG.info("Graph assembly (.pt output) complete: %d success, %d failures", edge_result["success"], len(edge_result["failures"]))
    LOG.info("Graph (.pt) generation elapsed time: %.2f s", edge_result.get("elapsed", 0.0))

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
    )
    LOG.info("Summary log written to %s", _relative_path(text_summary_path, run_info.root_dir))

    if edge_result["success"] == 0:
        LOG.error("No graphs produced. See logs for details.")
        return 1

    LOG.info("Dynamic graph builder completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
