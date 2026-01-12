#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAIN_CLI = "qtdaqa.new_dynamic_features.model_training2.train_cli"


@dataclasses.dataclass
class FeatureGroup:
    name: str
    indices: List[int]
    columns: List[str]
    kind: str = "topology"
    score: float = 0.0


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_graph_metadata(graph_dir: Path) -> dict:
    meta_path = graph_dir / "graph_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"graph_metadata.json not found in {graph_dir}")
    return _load_json(meta_path)


def _extract_node_columns(meta: dict) -> List[str]:
    node_schema = meta.get("_node_schema") or meta.get("node_schema") or {}
    if isinstance(node_schema, dict):
        columns = node_schema.get("columns")
        if isinstance(columns, list) and columns:
            return [str(c) for c in columns]
    raise ValueError("node schema columns not found in graph_metadata.json")


def _extract_topology_columns(meta: dict, graph_dir: Path) -> List[str]:
    topo_schema = meta.get("_topology_schema") or meta.get("topology_schema") or {}
    if isinstance(topo_schema, dict):
        columns = topo_schema.get("columns")
        if isinstance(columns, list) and columns:
            return [str(c) for c in columns]
    topo_path = graph_dir / "topology_columns.json"
    if topo_path.exists():
        columns = _load_json(topo_path)
        if isinstance(columns, list):
            return [str(c) for c in columns]
    return []


def _infer_slot_id(column: str) -> str:
    if column.startswith("base."):
        return "base"
    if column.startswith("cross."):
        return "cross"
    if column.startswith("polar."):
        return "polar"
    if column.startswith("primary."):
        parts = column.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    if column.startswith("pair."):
        parts = column.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    if column.startswith("typed_primary."):
        parts = column.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    if column.startswith("typed_pair."):
        parts = column.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    if column.startswith("strat."):
        parts = column.split(".")
        if len(parts) >= 4:
            return ".".join(parts[:4])
    if "." in column:
        return column.split(".", 1)[0]
    return column


def _group_topology_columns(topo_cols: Sequence[str], mode: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    if mode == "column":
        for col in topo_cols:
            groups[col] = [col]
        return groups
    for col in topo_cols:
        slot = _infer_slot_id(col)
        groups.setdefault(slot, []).append(col)
    return groups


def build_feature_groups(
    node_cols: Sequence[str],
    topo_cols: Sequence[str],
    *,
    include_dssp: bool,
    group_mode: str,
) -> List[FeatureGroup]:
    node_index = {name: idx for idx, name in enumerate(node_cols)}
    groups: List[FeatureGroup] = []
    topo_grouped = _group_topology_columns(topo_cols, group_mode)
    for name, cols in topo_grouped.items():
        indices = [node_index[c] for c in cols if c in node_index]
        if indices:
            groups.append(FeatureGroup(name=name, indices=indices, columns=list(cols)))
    if include_dssp:
        topo_set = set(topo_cols)
        dssp_cols = [c for c in node_cols if c not in topo_set]
        dssp_indices = [node_index[c] for c in dssp_cols]
        if dssp_indices:
            groups.append(
                FeatureGroup(name="dssp", indices=dssp_indices, columns=dssp_cols, kind="dssp")
            )
    if not groups:
        raise ValueError("No feature groups derived from metadata.")
    return groups


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint does not contain a state_dict.")
    return payload


def score_groups_from_checkpoint(groups: List[FeatureGroup], checkpoint_path: Path) -> List[FeatureGroup]:
    state = _load_state_dict(checkpoint_path)
    weight_keys = [
        k for k in state.keys() if k.endswith(".weight") and (".embed." in k or k.startswith("embed."))
    ]
    weights = [state[k].detach().cpu().numpy() for k in weight_keys]
    if not weights:
        raise ValueError("No embedding weights found in checkpoint for scoring.")
    feature_scores = np.zeros(weights[0].shape[1], dtype=np.float64)
    for weight in weights:
        feature_scores += np.abs(weight).sum(axis=0)
    for group in groups:
        group.score = float(np.sum(feature_scores[group.indices]))
    return groups


def make_mask(
    groups: List[FeatureGroup],
    *,
    keep_fraction: float,
    always_keep: Optional[Iterable[str]] = None,
) -> np.ndarray:
    keep_fraction = max(0.0, min(1.0, keep_fraction))
    sorted_groups = sorted(groups, key=lambda g: g.score, reverse=True)
    keep_count = max(1, int(math.ceil(len(sorted_groups) * keep_fraction)))
    keep = {g.name for g in sorted_groups[:keep_count]}
    if always_keep:
        keep.update(always_keep)
    length = max(idx for g in groups for idx in g.indices) + 1
    mask = np.zeros(length, dtype=np.float32)
    for group in groups:
        if group.name in keep:
            mask[group.indices] = 1.0
    return mask


def write_mask(path: Path, mask: np.ndarray, groups: List[FeatureGroup], keep_fraction: float) -> None:
    payload = {
        "mask": mask.tolist(),
        "keep_fraction": float(keep_fraction),
        "groups": [
            {
                "name": g.name,
                "kind": g.kind,
                "score": float(g.score),
                "size": len(g.indices),
            }
            for g in groups
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _encode_override(value: object) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip()


def _run_train_cli(
    *,
    config_path: Path,
    output_root: Path,
    run_name: str,
    overrides: Dict[str, object],
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        TRAIN_CLI,
        "run",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
    ]
    for key, value in overrides.items():
        cmd.extend(["--override", f"{key}={_encode_override(value)}"])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"train_cli failed for {run_name} (code={result.returncode})")
    run_dir = output_root / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Expected run dir not found: {run_dir}")
    return run_dir


def _resolve_checkpoint(run_dir: Path) -> Path:
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        meta = _load_json(meta_path)
        ckpt = None
        checkpoints = meta.get("checkpoints") if isinstance(meta, dict) else None
        if isinstance(checkpoints, dict):
            ckpt = checkpoints.get("primary_path") or checkpoints.get("primary_checkpoint")
        if ckpt:
            candidate = Path(str(ckpt))
            if candidate.exists():
                return candidate
    best_link = run_dir / "model_checkpoints" / "best.ckpt"
    if best_link.exists():
        return best_link
    candidates = sorted((run_dir / "model_checkpoints").glob("*.chkpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return candidates[0]


def _load_run_metrics(run_dir: Path) -> Dict[str, object]:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    meta = _load_json(meta_path)
    metrics = {}
    for key in (
        "best_val_loss",
        "best_selection_metric",
        "best_val_rank_spearman",
        "best_val_rank_regret",
        "best_val_spearman_corr",
        "best_val_rank_spearman",
    ):
        if key in meta:
            metrics[key] = meta[key]
    metrics["run_dir"] = str(run_dir)
    return metrics


def _write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    if rows:
        keys = sorted(rows[0].keys())
        lines = [",".join(keys)]
        for row in rows:
            lines.append(",".join(str(row.get(k, "")) for k in keys))
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    baseline = rows[0]
    lines = [
        "# Feature Lottery Ticket (FLT) Summary",
        "",
        "## How to interpret",
        "1. Use the baseline (keep_fraction=1.0) as your reference.",
        "2. Identify the smallest keep_fraction whose metrics stay within tolerance of the baseline.",
        "3. The corresponding kept groups form the feature lottery ticket.",
        "",
        "Recommended tolerance (publishable): within 1% relative change on val_loss and within 0.01 absolute on rank Spearman.",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        lines.append(f"- keep_fraction={row.get('keep_fraction')}, val_loss={row.get('best_val_loss')}, "
                     f"selection_metric={row.get('best_selection_metric')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_flt(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    graph_raw = args.graph_dir or base_config.get("paths", {}).get("graph") or base_config.get("graph_dir")
    if not graph_raw:
        raise ValueError("graph_dir not provided and not found in config.")
    graph_dir = Path(graph_raw)
    if not graph_dir.is_absolute():
        graph_dir = (config_path.parent / graph_dir).resolve()

    meta = _load_graph_metadata(graph_dir)
    node_cols = _extract_node_columns(meta)
    topo_cols = _extract_topology_columns(meta, graph_dir)
    groups = build_feature_groups(
        node_cols,
        topo_cols,
        include_dssp=not args.drop_dssp,
        group_mode=args.group_mode,
    )

    baseline_run = Path(args.baseline_run).resolve() if args.baseline_run else None
    init_state_path = None
    if baseline_run is None:
        baseline_run = _run_train_cli(
            config_path=config_path,
            output_root=output_root,
            run_name="flt_dense",
            overrides={"feature_lottery.save_init_state": True},
        )
    init_candidate = baseline_run / "model_checkpoints" / "init_state.pt"
    if init_candidate.exists():
        init_state_path = init_candidate

    checkpoint_path = _resolve_checkpoint(baseline_run)
    groups = score_groups_from_checkpoint(groups, checkpoint_path)

    scores_path = output_root / "flt_group_scores.json"
    scores_payload = [
        {"name": g.name, "kind": g.kind, "score": g.score, "size": len(g.indices)} for g in groups
    ]
    scores_path.write_text(json.dumps(scores_payload, indent=2), encoding="utf-8")

    rows: List[Dict[str, object]] = []
    baseline_metrics = _load_run_metrics(baseline_run)
    baseline_metrics["keep_fraction"] = 1.0
    rows.append(baseline_metrics)

    for frac in args.keep_fracs:
        if frac >= 1.0:
            continue
        mask = make_mask(groups, keep_fraction=frac, always_keep={"dssp"} if not args.drop_dssp else None)
        mask_path = output_root / f"flt_mask_keep_{frac:.3f}.json"
        write_mask(mask_path, mask, groups, frac)
        overrides: Dict[str, object] = {
            "feature_mask.path": str(mask_path),
            "feature_mask.apply_to": "node",
            "feature_mask.strict": True,
        }
        if init_state_path is not None:
            overrides["feature_lottery.init_state_path"] = str(init_state_path)
        run_dir = _run_train_cli(
            config_path=config_path,
            output_root=output_root,
            run_name=f"flt_keep_{frac:.3f}",
            overrides=overrides,
        )
        metrics = _load_run_metrics(run_dir)
        metrics["keep_fraction"] = frac
        rows.append(metrics)

    summary_path = output_root / "flt_summary.json"
    _write_summary(summary_path, rows)
    _write_report(output_root / "flt_report.md", rows)


def _parse_keep_fracs(raw: str) -> List[float]:
    values = []
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        values.append(float(chunk))
    if 1.0 not in values:
        values.insert(0, 1.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Lottery Ticket (FLT) runner.")
    parser.add_argument("--config", required=True, help="Training config YAML.")
    parser.add_argument("--graph-dir", help="Override graph_dir (defaults to config paths.graph).")
    parser.add_argument("--output-dir", required=True, help="Output directory for FLT runs/artifacts.")
    parser.add_argument("--baseline-run", help="Existing baseline run dir (skip dense training).")
    parser.add_argument("--keep-fracs", default="1.0,0.7,0.5,0.3", help="Comma-separated keep fractions.")
    parser.add_argument("--group-mode", choices=("slot", "column"), default="slot", help="Grouping mode.")
    parser.add_argument("--drop-dssp", action="store_true", help="Do not force-keep DSSP features.")
    args = parser.parse_args()
    args.keep_fracs = _parse_keep_fracs(args.keep_fracs)
    run_flt(args)


if __name__ == "__main__":
    main()
