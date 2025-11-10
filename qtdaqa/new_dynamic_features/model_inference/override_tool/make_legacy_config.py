#!/usr/bin/env python3
"""Generate an inference config skeleton for a legacy checkpoint.

This CLI inspects `graph_metadata.json` plus sample `.pt` graphs using the same
`load_graph_feature_metadata` logic as training/inference. It aggregates the schema
exactly the way the downstream stack does (edge/node/topology), then emits a config
with `use_checkpoint_schema=false` and a populated overrides block so you don’t have
to hand-enter feature definitions. If the metadata truly isn’t present, the script
adds a note reminding you to fill the schema manually. You still need to edit the
placeholder paths under `paths.*` (data_dir, work_dir, output_file, label_file) before
running `run_model_inference.sh`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.common.feature_metadata import (  # noqa: E402
    GraphFeatureMetadata,
    load_graph_feature_metadata,
)


def _serialise_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in schema.items() if value not in (None, {}, [], "")}


def _serialise_module(module_entry: Any) -> Dict[str, Any]:
    if not isinstance(module_entry, dict):
        return {}
    allowed_keys = {"id", "module", "alias", "summary", "jobs", "defaults", "parameters"}
    return {key: value for key, value in module_entry.items() if key in allowed_keys and value not in (None, {}, [], "")}


def build_config(graph_dir: Path, checkpoint: Path) -> Dict[str, Any]:
    metadata: GraphFeatureMetadata = load_graph_feature_metadata(graph_dir)

    overrides: Dict[str, Dict[str, Any]] = {}
    edge_schema = _serialise_schema(metadata.edge_schema)
    node_schema = _serialise_schema(metadata.node_schema)
    topo_schema = _serialise_module(metadata.module_registry.get("topology"))
    if edge_schema:
        overrides["edge_schema"] = edge_schema
    if node_schema:
        overrides["node_schema"] = node_schema
    if topo_schema:
        overrides["topology_schema"] = topo_schema

    config: Dict[str, Any] = {
        "paths": {
            "data_dir": None,
            "work_dir": None,
            "output_file": None,
            "label_file": None,
            "checkpoint": str(checkpoint.resolve()),
            "training_root": None,
        },
        "builder": {
            "jobs": 6,
            "feature_config": None,
        },
        "options": {
            "reuse_existing_graphs": False,
            "use_checkpoint_schema": False,
        },
        "overrides": overrides if overrides else {"note": "No schema found; fill edge/node details manually."},
    }
    return config


class DescriptiveParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - argparse plumbing
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    description = (
        "Inspect graph_metadata.json and sample .pt files (via load_graph_feature_metadata) to build an "
        "inference config for a legacy checkpoint. The overrides block mirrors the real edge/node/topology "
        "schema, use_checkpoint_schema is set to false automatically, and you just need to fill in the "
        "placeholder paths before running run_model_inference.sh."
    )
    parser = DescriptiveParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.epilog = description
    parser.add_argument(
        "--graph-dir",
        required=True,
        type=Path,
        help="Path to the graph_data directory produced by graph_builder (must contain graph_metadata.json).",
    )
    parser.add_argument(
        "--legacy-checkpoint",
        required=True,
        type=Path,
        help="Legacy model checkpoint (.ckpt/.chkpt) that lacks embedded feature metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config.legacy.yaml"),
        help="Where to write the generated config file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = build_config(args.graph_dir, args.legacy_checkpoint)
    yaml_dump = yaml.safe_dump(config, sort_keys=False)
    args.output.write_text(yaml_dump)
    print(f"Wrote legacy inference config to {args.output}")
    print("Edit the placeholder paths (data_dir/work_dir/output_file/label_file) before running run_model_inference.sh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
