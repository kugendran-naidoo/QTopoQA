#!/usr/bin/env python3
"""Inference runner for the 24-D edge TopoQA model."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent

from . import builder_runner
from .builder_runner import (
    BuilderConfig,
    ensure_graph_dir,
    parse_builder_config,
    _preflight_registry_support,
    validate_graph_metadata,
)
from qtdaqa.new_dynamic_features.model_training import train_cli

if TYPE_CHECKING:  # pragma: no cover
    from qtdaqa.new_dynamic_features.model_training.model.gat_5_edge1 import (
        GNN_edge1_edgepooling,
    )


@dataclasses.dataclass
class InferenceConfig:
    data_dir: Optional[Path]
    work_dir: Optional[Path]
    checkpoint_path: Path
    results_dir: Optional[Path]
    output_file: Optional[Path]
    label_file: Optional[Path] = None
    batch_size: int = 32
    num_workers: int = 0
    builder: BuilderConfig = dataclasses.field(default_factory=BuilderConfig)
    reuse_existing_graphs: bool = False
    use_checkpoint_schema: bool = True
    force_node_dim: Optional[int] = None
    interface_schema: Dict[str, object] = dataclasses.field(default_factory=dict)
    topology_schema: Dict[str, object] = dataclasses.field(default_factory=dict)
    node_schema: Dict[str, object] = dataclasses.field(default_factory=dict)
    edge_schema: Dict[str, object] = dataclasses.field(default_factory=dict)
    training_root: Optional[Path] = None
    config_name: Optional[str] = None
    dataset_name: Optional[str] = None
    work_dir_base: Optional[Path] = None
    results_dir_base: Optional[Path] = None
    check_schema: bool = False


@dataclasses.dataclass
class LabelInfo:
    scores: Dict[str, float]
    targets: Dict[str, str]


def _normalise_model_name(name: str) -> str:
    trimmed = name.strip()
    return trimmed[:-5] if trimmed.endswith("_tidy") else trimmed


def _resolve(path_str: str | Path, base: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base / path).resolve()


def extract_feature_metadata(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("feature_metadata", {})


def _adapt_legacy_edge_embed_weights(
    state_dict: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    legacy_indices: List[str] = []
    for key in state_dict.keys():
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "edge_embed" and parts[1].isdigit() and parts[2] in {"weight", "bias"}:
            legacy_indices.append(parts[1])
    if not legacy_indices:
        return state_dict
    adapted = dict(state_dict)
    moved = False
    for idx in set(legacy_indices):
        for name in ("weight", "bias"):
            legacy_key = f"edge_embed.{idx}.{name}"
            new_key = f"edge_embed.{idx}.0.{name}"
            if legacy_key in adapted and new_key in target_state:
                adapted[new_key] = adapted.pop(legacy_key)
                moved = True
    if moved:
        logging.info(
            "Detected legacy edge_embed weights; remapped %d module(s) to the new encoder structure.", len(set(legacy_indices))
        )
    return adapted


def _uses_legacy_edge_encoder(state_dict: Dict[str, torch.Tensor]) -> bool:
    has_modern = False
    has_flat_linear = False
    for key in state_dict.keys():
        if not key.startswith("edge_embed."):
            continue
        parts = key.split(".")
        if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
            has_modern = True
            break
        if len(parts) == 3 and parts[1].isdigit() and parts[2] in {"weight", "bias"}:
            has_flat_linear = True
    return has_flat_linear and not has_modern


def resolve_feature_schema(cfg: InferenceConfig, checkpoint_meta: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    edge_schema: Dict[str, object] = {}
    topo_schema: Dict[str, object] = {}
    node_schema: Dict[str, object] = {}
    if cfg.use_checkpoint_schema:
        edge_schema.update(checkpoint_meta.get("edge_schema") or {})
        topo_schema.update(checkpoint_meta.get("topology_schema") or {})
        node_schema.update(checkpoint_meta.get("node_schema") or {})
    edge_schema.update(cfg.edge_schema or {})
    topo_schema.update(cfg.topology_schema or {})
    node_schema.update(cfg.node_schema or {})
    if cfg.force_node_dim is not None:
        node_schema["dim"] = int(cfg.force_node_dim)
    if not edge_schema:
        raise ValueError("Edge schema unavailable. Provide one in the checkpoint or config.yaml")
    final: Dict[str, Dict[str, object]] = {"edge_schema": edge_schema, "topology_schema": topo_schema}
    if node_schema:
        final["node_schema"] = node_schema
    if cfg.interface_schema:
        final["interface_schema"] = cfg.interface_schema
    return final


def _guard_schema_overrides(cfg: InferenceConfig, checkpoint_meta: Dict[str, object]) -> None:
    if not cfg.use_checkpoint_schema:
        return
    def _check(section: str, override: Dict[str, object], metadata: Dict[str, object]) -> None:
        if not override:
            return
        for key, expected in override.items():
            actual = metadata.get(key)
            if actual != expected:
                raise RuntimeError(
                    f"{section}.{key} mismatch: config override {expected!r} does not match checkpoint metadata {actual!r}. "
                    "Pick a checkpoint built with the same feature config or update the override."
                )

    edge_override = cfg.edge_schema or {}
    topo_override = cfg.topology_schema or {}
    edge_meta = checkpoint_meta.get("edge_schema") or {}
    topo_meta = checkpoint_meta.get("topology_schema") or {}
    node_meta = checkpoint_meta.get("node_schema") or {}
    _check("edge_schema", edge_override, edge_meta)
    _check("topology_schema", topo_override, topo_meta)
    _check("node_schema", cfg.node_schema or {}, node_meta)


def _default_training_root() -> Path:
    return SCRIPT_DIR.parent / "model_training" / "training_runs"


def _auto_select_checkpoint(training_root: Path) -> Path:
    root = training_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Training root does not exist: {root}")

    candidates: List[Tuple[float, Path, str]] = []
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_symlink() or not run_dir.is_dir():
            continue
        try:
            summary = train_cli._summarise_run(run_dir)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Failed to summarise training run %s: %s", run_dir, exc)
            continue

        checkpoint = summary.get("best_checkpoint")
        if not checkpoint:
            continue
        selection_metric = summary.get("best_selection_metric")
        val_loss = summary.get("best_val_loss")
        if selection_metric is None and val_loss is None:
            continue
        score = float(selection_metric if selection_metric is not None else val_loss)
        candidates.append((score, Path(checkpoint), summary.get("run_name", run_dir.name)))

    if not candidates:
        raise RuntimeError(f"No eligible checkpoints found under {root}")

    candidates.sort(key=lambda item: item[0])
    best_score, checkpoint_path, run_name = candidates[0]
    resolved = checkpoint_path.resolve()
    logging.info(
        "Auto-selected checkpoint %s (score=%s, run=%s) from training root %s",
        resolved,
        best_score,
        run_name,
        root,
    )
    return resolved


def _log_checkpoint_banner(cfg: InferenceConfig, surround_blank: bool = False) -> None:
    if surround_blank:
        logging.info("")
    logging.info("Checkpoint file: %s", cfg.checkpoint_path)
    if cfg.config_name:
        logging.info("Inference config: %s", cfg.config_name)
    if surround_blank:
        logging.info("")


def _write_inference_schema_summary(
    cfg: InferenceConfig,
    graph_metadata: "GraphFeatureMetadata",
    final_schema: Dict[str, Dict[str, object]],
    checkpoint_meta: Dict[str, object],
) -> Path:
    metadata_dict = graph_metadata.to_dict()
    summary_payload = {
        "checkpoint": str(cfg.checkpoint_path),
        "config_name": cfg.config_name,
        "data_dir": str(cfg.data_dir),
        "work_dir": str(cfg.work_dir),
        "results_dir": str(cfg.results_dir),
        "output_file": str(cfg.output_file),
        "label_file": str(cfg.label_file) if cfg.label_file else None,
        "training_root": str(cfg.training_root) if cfg.training_root else None,
        "use_checkpoint_schema": cfg.use_checkpoint_schema,
        "dataset_name": cfg.dataset_name,
        "overrides": {
            "interface_schema": cfg.interface_schema,
            "node_schema": cfg.node_schema,
            "edge_schema": cfg.edge_schema,
            "topology_schema": cfg.topology_schema,
        },
        "graph_metadata": metadata_dict,
        "final_schema": final_schema,
        "checkpoint_schema": {
            "edge_schema": checkpoint_meta.get("edge_schema"),
            "topology_schema": checkpoint_meta.get("topology_schema"),
        },
    }
    summary = {
        "checkpoint": summary_payload["checkpoint"],
        "config_name": summary_payload["config_name"],
        "data_dir": summary_payload["data_dir"],
        "work_dir": summary_payload["work_dir"],
        "results_dir": summary_payload["results_dir"],
        "output_file": summary_payload["output_file"],
        "label_file": summary_payload["label_file"],
        "training_root": summary_payload["training_root"],
        "use_checkpoint_schema": summary_payload["use_checkpoint_schema"],
        "dataset_name": summary_payload["dataset_name"],
        "overrides": summary_payload["overrides"],
        "final_schema": summary_payload["final_schema"],
        "checkpoint_schema": summary_payload["checkpoint_schema"],
        "graph_metadata": summary_payload["graph_metadata"],
    }
    summary_path = cfg.work_dir / "schema_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def _run_schema_check(cfg: InferenceConfig, final_schema: Dict[str, Dict[str, object]]) -> None:
    logging.info("Schema check-only mode enabled; no graphs will be built.")
    builder_runner._preflight_registry_support(final_schema)

    graph_dir = None
    if cfg.work_dir:
        graph_dir = Path(cfg.work_dir) / "graph_data"

    if graph_dir and graph_dir.exists() and any(graph_dir.rglob("*.pt")):
        logging.info("Validating existing graphs at %s against the resolved schema...", graph_dir)
        builder_runner.validate_graph_metadata(graph_dir, final_schema)
        logging.info("Existing graphs are compatible with the resolved schema.")
    else:
        if graph_dir:
            logging.info(
                "No existing graphs found under %s. Schema is supported locally; build graphs before running inference.",
                graph_dir,
            )
        else:
            logging.info("No work_dir provided; schema is supported locally.")


def load_config(raw_path: Path) -> InferenceConfig:
    with raw_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Inference configuration must be a mapping.")

    legacy_keys = {
        "data_dir",
        "work_dir",
        "output_file",
        "results_dir",
        "dataset_name",
        "checkpoint_path",
        "label_file",
        "reuse_existing_graphs",
        "use_checkpoint_schema",
    }
    legacy_present = sorted(legacy_keys.intersection(data))
    if legacy_present:
        raise ValueError(
            "Inference config must use the structured layout with 'paths', 'builder', and 'options' sections. "
            f"Legacy top-level keys detected: {', '.join(legacy_present)}."
        )

    paths_cfg = data.get("paths") or {}
    if not isinstance(paths_cfg, dict):
        raise ValueError("Inference configuration must contain a 'paths' mapping.")

    def _require_or_none(key: str) -> Optional[object]:
        return paths_cfg.get(key)

    data_dir_raw = _require_or_none("data_dir")
    work_dir_raw = _require_or_none("work_dir")
    results_dir_raw = _require_or_none("results_dir")
    dataset_name_raw = paths_cfg.get("dataset_name")
    checkpoint_raw = paths_cfg.get("checkpoint")
    training_root_raw = paths_cfg.get("training_root")
    data_dir = _resolve(data_dir_raw, raw_path.parent) if data_dir_raw else None
    work_dir = _resolve(work_dir_raw, raw_path.parent) if work_dir_raw else None
    results_dir = _resolve(results_dir_raw, raw_path.parent) if results_dir_raw else None
    label_file_raw = paths_cfg.get("label_file")
    label_file = _resolve(label_file_raw, raw_path.parent) if label_file_raw else None
    training_root = (
        _resolve(training_root_raw, raw_path.parent)
        if training_root_raw
        else _default_training_root()
    )

    builder_cfg = parse_builder_config(data.get("builder"), raw_path.parent)
    options_cfg = data.get("options") or {}
    if options_cfg and not isinstance(options_cfg, dict):
        raise ValueError("options section must be a mapping when provided.")
    reuse_existing = bool(options_cfg.get("reuse_existing_graphs", False)) if options_cfg else False
    use_checkpoint_schema = options_cfg.get("use_checkpoint_schema", True)
    use_checkpoint_schema = bool(use_checkpoint_schema)
    check_schema = bool(options_cfg.get("check_schema", False)) if options_cfg else False
    force_node_dim = options_cfg.get("force_node_dim")
    if force_node_dim is not None:
        try:
            force_node_dim = int(force_node_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError("options.force_node_dim must be an integer when provided.") from exc
        if force_node_dim <= 0:
            raise ValueError("options.force_node_dim must be greater than zero.")

    interface_schema_cfg = data.get("interface_schema") or {}
    if interface_schema_cfg and not isinstance(interface_schema_cfg, dict):
        raise ValueError("interface_schema section must be a mapping when provided.")
    topology_schema_cfg = data.get("topology_schema") or {}
    if topology_schema_cfg and not isinstance(topology_schema_cfg, dict):
        raise ValueError("topology_schema section must be a mapping when provided.")
    node_schema_cfg = data.get("node_schema") or {}
    if node_schema_cfg and not isinstance(node_schema_cfg, dict):
        raise ValueError("node_schema section must be a mapping when provided.")
    edge_schema_cfg = data.get("edge_schema") or {}
    if edge_schema_cfg and not isinstance(edge_schema_cfg, dict):
        raise ValueError("edge_schema section must be a mapping when provided.")

    if checkpoint_raw:
        checkpoint_path = _resolve(checkpoint_raw, raw_path.parent)
    else:
        if not training_root.exists():
            raise FileNotFoundError(
                f"Training root does not exist: {training_root}. "
                "Set paths.training_root in your inference config or point at the correct training_runs directory."
            )
        checkpoint_path = _auto_select_checkpoint(training_root)

    dataset_name = str(dataset_name_raw).strip() if dataset_name_raw else None

    return InferenceConfig(
        data_dir=data_dir,
        work_dir=None,
        checkpoint_path=checkpoint_path,
        results_dir=None,
        output_file=None,
        label_file=label_file,
        batch_size=int(data.get("batch_size", 32)),
        num_workers=int(data.get("num_workers", 0)),
        builder=builder_cfg,
        reuse_existing_graphs=reuse_existing,
        use_checkpoint_schema=use_checkpoint_schema,
        force_node_dim=force_node_dim,
        interface_schema=interface_schema_cfg or {},
        topology_schema=topology_schema_cfg or {},
        node_schema=node_schema_cfg or {},
        edge_schema=edge_schema_cfg,
        training_root=training_root,
        config_name=raw_path.name,
        dataset_name=dataset_name,
        work_dir_base=work_dir,
        results_dir_base=results_dir,
        check_schema=check_schema,
    )


class GraphInferenceDataset(Dataset):
    def __init__(
        self,
        graph_items: Sequence[Tuple[str, Path]],
        label_map: Optional[Dict[str, float]] = None,
        target_map: Optional[Dict[str, str]] = None,
    ):
        self.graph_items = [(key, Path(path)) for key, path in graph_items]
        self.label_map = label_map or {}
        self.target_map = target_map or {}

    def __len__(self) -> int:
        return len(self.graph_items)

    def __getitem__(self, idx: int) -> Data:
        model_key, path = self.graph_items[idx]
        data = torch.load(path)

        if hasattr(data, "batch"):
            batch_attr = getattr(data, "batch")
            if batch_attr is None or not torch.is_tensor(batch_attr):
                delattr(data, "batch")

        base_name = Path(model_key).name
        normalised_name = _normalise_model_name(base_name)
        data.name = base_name
        data.model_key = model_key

        def _ordered_candidates() -> List[str]:
            ordered: List[str] = []
            seen = set()

            def _add(entry: Optional[str]) -> None:
                if entry and entry not in seen:
                    ordered.append(entry)
                    seen.add(entry)

            _add(model_key)
            if "/" in model_key:
                prefix, _, _ = model_key.partition("/")
                _add(f"{prefix}/{normalised_name}")
                _add(f"{prefix}/{base_name}")
            _add(base_name)
            _add(normalised_name)
            return ordered

        candidate_keys = _ordered_candidates()

        target_value = None
        for candidate in candidate_keys:
            if candidate in self.target_map and self.target_map[candidate] is not None:
                target_value = self.target_map[candidate]
                break
        if target_value is None and "/" in model_key:
            target_value = model_key.split("/", 1)[0]
        if target_value is not None:
            data.target = str(target_value)

        if self.label_map:
            label_value = None
            for candidate in candidate_keys:
                if candidate in self.label_map:
                    label_value = self.label_map[candidate]
                    break
            if label_value is not None:
                data.y = torch.tensor([label_value], dtype=torch.float32)
            else:
                data.y = torch.tensor([float("nan")], dtype=torch.float32)
        return data


def _ensure_list(value: object) -> List[object]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.view(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def collate_graphs(batch: List[Data]) -> Batch:
    if not batch:
        raise ValueError("Empty batch passed to inference loader")
    merged = Batch.from_data_list(batch)
    merged.name = [str(getattr(item, "name", "")) for item in batch]
    if any(hasattr(item, "target") for item in batch):
        merged.target = [getattr(item, "target", None) for item in batch]
    if any(hasattr(item, "model_key") for item in batch):
        merged.model_key = [getattr(item, "model_key", None) for item in batch]
    return merged


def load_label_map(label_path: Path) -> LabelInfo:
    import pandas as pd

    df = pd.read_csv(label_path)
    normalized = {col.strip().lower(): col for col in df.columns}
    model_col = normalized.get("model")
    dockq_col = normalized.get("dockq")
    if not model_col or not dockq_col:
        raise ValueError(f"Label CSV {label_path} missing MODEL/dockq columns")
    target_col = normalized.get("target")

    model_series = df[model_col].astype(str)
    dockq_series = pd.to_numeric(df[dockq_col], errors="coerce")
    targets: Dict[str, str] = {}
    scores: Dict[str, float] = {}

    target_series = df[target_col] if target_col else None

    conflict_targets: Dict[str, set] = defaultdict(set)

    def _record_score(key: str, value: float, target: Optional[str]) -> None:
        if "/" not in key:
            existing = scores.get(key)
            if existing is not None and not math.isclose(existing, value, rel_tol=1e-9, abs_tol=1e-9):
                conflict_targets[key].add(target or "<unknown>")
        scores[key] = value

    for idx, model_value in enumerate(model_series):
        raw_name = model_value.strip()
        norm_name = _normalise_model_name(raw_name)
        base_keys = {raw_name, norm_name}
        base_keys.discard("")

        target_value = None
        if target_series is not None:
            raw_target = target_series.iloc[idx]
            if pd.notna(raw_target):
                target_value = str(raw_target).strip()

        composite_keys = set()
        if target_value:
            for key in base_keys:
                if key:
                    composite_keys.add(f"{target_value}/{key}")

        dockq_value = dockq_series.iloc[idx]
        if pd.notna(dockq_value):
            value = float(dockq_value)
            for key in composite_keys | base_keys:
                if key:
                    _record_score(key, value, target_value)

        if target_value:
            for key in composite_keys:
                targets[key] = target_value
            for key in base_keys:
                targets.setdefault(key, target_value)

    if conflict_targets:
        preview = ", ".join(
            f"{key} -> {', '.join(sorted(values))}"
            for key, values in list(conflict_targets.items())[:5]
        )
        logging.warning(
            "Label CSV %s contains duplicate unqualified model names mapped to multiple targets (examples: %s). "
            "Ensure per-target names remain unique or rely on target-qualified keys.",
            label_path,
            preview,
        )

    return LabelInfo(scores=scores, targets=targets)


def _format_csv_float(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    text = f"{number:.8f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _format_truth_float(value: object) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    rounded = round(number, 3)
    text = f"{rounded:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _write_target_reports(df: "pd.DataFrame", output_root: Path) -> None:
    import pandas as pd

    if "TARGET" not in df.columns:
        return
    target_df = df.dropna(subset=["TARGET"])
    if target_df.empty:
        return

    for target_value, group in target_df.groupby("TARGET", sort=True):
        target_dir = output_root / str(target_value)
        target_dir.mkdir(parents=True, exist_ok=True)

        group_sorted = group.sort_values("PRED_DOCKQ", ascending=False)
        result_df = group_sorted[["MODEL", "PRED_DOCKQ"]].copy()
        result_df["PRED_DOCKQ"] = result_df["PRED_DOCKQ"].map(_format_csv_float)
        result_df.to_csv(target_dir / "result.csv", index=False)

        if "DockQ" not in group_sorted.columns:
            continue

        truth_group = group_sorted.dropna(subset=["DockQ"]).copy()
        if truth_group.empty:
            continue

        truth_group = truth_group.assign(
            TARGET=str(target_value),
            MODEL_CLEAN=truth_group["MODEL"].map(_normalise_model_name),
        )
        unified_df = truth_group[["TARGET", "MODEL_CLEAN", "PRED_DOCKQ", "DockQ"]].rename(
            columns={"MODEL_CLEAN": "MODEL", "DockQ": "TRUE_DOCKQ"}
        )
        unified_df["PRED_DOCKQ"] = unified_df["PRED_DOCKQ"].astype(float)
        unified_df["TRUE_DOCKQ"] = unified_df["TRUE_DOCKQ"].astype(float)

        unified_path = target_dir / f"{target_value}.unified_result.csv"
        unified_to_write = unified_df.copy()
        unified_to_write["PRED_DOCKQ"] = unified_to_write["PRED_DOCKQ"].map(_format_csv_float)
        unified_to_write["TRUE_DOCKQ"] = unified_to_write["TRUE_DOCKQ"].map(_format_truth_float)
        unified_to_write.to_csv(unified_path, index=False)

        best_true_idx = unified_df["TRUE_DOCKQ"].astype(float).idxmax()
        best_true_row = unified_df.loc[best_true_idx]
        top_pred_df = unified_df.sort_values("PRED_DOCKQ", ascending=False)
        top_pred_row = top_pred_df.iloc[0]

        m_star = float(best_true_row["TRUE_DOCKQ"])
        top_pred_true = float(top_pred_row["TRUE_DOCKQ"])
        top_pred_pred = float(top_pred_row["PRED_DOCKQ"])
        ranking_loss_value = m_star - top_pred_true

        top10_df = top_pred_df.head(10).copy()
        hit_counts = [
            int((top10_df["TRUE_DOCKQ"] >= threshold).sum())
            for threshold in (0.23, 0.49, 0.80)
        ]

        metrics_path = target_dir / f"{target_value}.summary_metrics.csv"
        with metrics_path.open("w", encoding="utf-8") as handle:
            handle.write("Note last 4 columns ranking_loss, top10_A, top10_B, top10_C\n\n")
            handle.write("TARGET,m_star_best_true,top_pred_true,top_pred_pred,ranking_loss,top10_A,top10_B,top10_C\n")
            handle.write(
                f"{target_value},{m_star:.6f},{top_pred_true:.6f},{top_pred_pred:.6f},{ranking_loss_value:.6f},"
                f"{hit_counts[0]},{hit_counts[1]},{hit_counts[2]}\n"
            )

        top10_lines = [
            f"{target_value},{row['MODEL']},{_format_csv_float(row['PRED_DOCKQ'])},{_format_truth_float(row['TRUE_DOCKQ'])}"
            for _, row in top10_df.iterrows()
        ]
        hit_path = target_dir / f"{target_value}.hit.rate_result.csv"
        with hit_path.open("w", encoding="utf-8") as handle:
            handle.write("Acceptable-or-better (a): count DockQ ≥ 0.23\n")
            handle.write("Medium-or-better (b): count DockQ ≥ 0.49\n")
            handle.write("High (c): count DockQ ≥ 0.80\n\n")
            handle.write("hit rate = a/b/c \n\n")
            handle.write(
                f"{target_value} hit rate =  {hit_counts[0]}/{hit_counts[1]}/{hit_counts[2]}\n\n"
            )
            handle.write("\n".join(top10_lines))
            handle.write("\n")

        ranking_path = target_dir / f"{target_value}.ranking_loss_result.csv"
        m_star_disp = f"{m_star:.3f}"
        m_hat_disp = f"{top_pred_true:.3f}"
        rl_disp = f"{ranking_loss_value:.3f}"
        best_true_line = (
            f"{target_value},{best_true_row['MODEL']},{_format_csv_float(best_true_row['PRED_DOCKQ'])},"
            f"{_format_truth_float(best_true_row['TRUE_DOCKQ'])}"
        )
        top_pred_line = (
            f"{target_value},{top_pred_row['MODEL']},{_format_csv_float(top_pred_row['PRED_DOCKQ'])},"
            f"{_format_truth_float(top_pred_row['TRUE_DOCKQ'])}"
        )
        with ranking_path.open("w", encoding="utf-8") as handle:
            handle.write("ranking loss = m* - m^\n")
            handle.write(f"{target_value} m* = {m_star_disp}\n")
            handle.write(f"{target_value} m^ = {m_hat_disp}\n")
            handle.write(f"{target_value} ranking loss = {rl_disp}\n\n")
            handle.write(best_true_line + "\n")
            handle.write(top_pred_line + "\n")


def gather_graphs(graph_dir: Path) -> List[Tuple[str, Path]]:
    paths = sorted(graph_dir.rglob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt graphs found under {graph_dir}")
    entries: List[Tuple[str, Path]] = []
    for path in paths:
        try:
            relative = path.relative_to(graph_dir)
        except ValueError:
            key = path.stem
        else:
            key = relative.with_suffix("").as_posix()
        entries.append((key, path))
    return entries


def load_model(cfg: InferenceConfig, edge_schema: Dict[str, object], node_schema: Dict[str, object]):
    from qtdaqa.new_dynamic_features.model_training.model.gat_5_edge1 import (  # noqa: WPS433
        GNN_edge1_edgepooling,
    )

    edge_dim = int(edge_schema.get("dim", 24))
    node_dim = node_schema.get("dim")
    if node_dim is None:
        raise RuntimeError("Node schema is missing 'dim'; cannot construct model. Ensure checkpoint or config supplies node_dim.")
    try:
        node_dim_int = int(node_dim)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Node schema 'dim' must be an integer, got {node_dim!r}") from exc
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    edge_encoder_variant = edge_schema.get("encoder_variant")
    if not edge_encoder_variant:
        edge_encoder_variant = "legacy_linear" if _uses_legacy_edge_encoder(state_dict) else "modern"

    model = GNN_edge1_edgepooling(
        init_lr=1.0,
        pooling_type="mean",
        mode="zuhe",
        num_net=1,
        hidden_dim=32,
        edge_dim=edge_dim,
        output_dim=64,
        n_output=1,
        heads=8,
        edge_schema=edge_schema,
        node_dim=node_dim_int,
        edge_encoder_variant=edge_encoder_variant,
    )
    if edge_encoder_variant != "legacy_linear":
        state_dict = _adapt_legacy_edge_embed_weights(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    model.eval()
    return model


def run_inference(
    cfg: InferenceConfig,
    final_schema: Dict[str, Dict[str, object]],
    feature_metadata: Dict[str, object],
) -> None:
    if cfg.work_dir is None or cfg.results_dir is None or cfg.output_file is None:
        raise RuntimeError("Inference configuration missing required directories; ensure work_dir/results_dir/dataset_name are set.")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    _log_checkpoint_banner(cfg, surround_blank=True)
    graph_dir = ensure_graph_dir(cfg, final_schema, feature_metadata)
    logging.info("Graph directory: %s", graph_dir)
    _log_checkpoint_banner(cfg, surround_blank=False)
    metadata = validate_graph_metadata(graph_dir, final_schema)
    metadata_source = metadata.metadata_path or str(graph_dir / "graph_metadata.json")
    logging.info("Verified graph metadata compatibility (%s).", metadata_source)
    summary_path = _write_inference_schema_summary(cfg, metadata, final_schema, feature_metadata)
    logging.info("Inference schema summary written to %s", summary_path)

    logging.info("Loading graphs from %s", graph_dir)
    graph_entries = gather_graphs(graph_dir)
    label_info = load_label_map(cfg.label_file) if cfg.label_file else None

    dataset = GraphInferenceDataset(
        graph_entries,
        label_info.scores if label_info else None,
        label_info.targets if label_info else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_graphs,
        persistent_workers=False,
    )

    node_schema = final_schema.get("node_schema") or {}
    model = load_model(cfg, final_schema["edge_schema"], node_schema)

    results: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            scores = model([batch]).view(-1).cpu().tolist()
            name_attr = getattr(batch, "name", None)
            key_attr = getattr(batch, "model_key", None)
            target_attr = getattr(batch, "target", None)
            truth_attr = getattr(batch, "y", None)

            batch_names = [str(n) for n in _ensure_list(name_attr)]
            batch_keys = [str(k) for k in _ensure_list(key_attr)]
            batch_targets = [t if (t is None or isinstance(t, str)) else str(t) for t in _ensure_list(target_attr)]

            truth_values: List[Optional[float]] = []
            for item in _ensure_list(truth_attr):
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    truth_values.append(None)
                else:
                    truth_values.append(None if math.isnan(value) else value)

            for idx_in_batch, pred in enumerate(scores):
                key = batch_keys[idx_in_batch] if idx_in_batch < len(batch_keys) else None
                if key is None:
                    key = batch_names[idx_in_batch] if idx_in_batch < len(batch_names) else f"sample_{len(results) + idx_in_batch}"
                base_name = batch_names[idx_in_batch] if idx_in_batch < len(batch_names) else Path(key).name

                target_value: Optional[str] = None
                if idx_in_batch < len(batch_targets):
                    candidate = batch_targets[idx_in_batch]
                    if candidate not in (None, ""):
                        target_value = str(candidate)
                if label_info and label_info.targets:
                    target_value = target_value or label_info.targets.get(key)
                    if target_value is None and "/" in key:
                        prefix, _, suffix = key.partition("/")
                        candidates = [
                            f"{prefix}/{suffix}",
                            f"{prefix}/{_normalise_model_name(suffix)}",
                        ]
                        for candidate_key in candidates:
                            if candidate_key in label_info.targets:
                                target_value = label_info.targets[candidate_key]
                                break
                if target_value is None and "/" in key:
                    target_value = key.split("/", 1)[0]

                true_value = truth_values[idx_in_batch] if idx_in_batch < len(truth_values) else None

                results.append(
                    {
                        "model_key": key,
                        "model": base_name,
                        "target": target_value,
                        "pred": float(pred),
                        "true": true_value,
                    }
                )

    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for item in results:
        row: Dict[str, object] = {}
        if item["target"] is not None:
            row["TARGET"] = str(item["target"])
        row["MODEL_KEY"] = item["model_key"]
        row["MODEL"] = item["model"]
        row["PRED_DOCKQ"] = item["pred"]
        if item["true"] is not None:
            row["DockQ"] = item["true"]
        rows.append(row)

    import pandas as pd

    df = pd.DataFrame(rows)
    if not df.empty:
        column_order: List[str] = []
        if "TARGET" in df.columns:
            column_order.append("TARGET")
        if "MODEL_KEY" in df.columns:
            column_order.append("MODEL_KEY")
        for column in ("MODEL", "PRED_DOCKQ"):
            if column in df.columns:
                column_order.append(column)
        if "DockQ" in df.columns:
            column_order.append("DockQ")
        df = df[column_order]
    _write_target_reports(df, cfg.output_file.parent)
    df.to_csv(cfg.output_file, index=False)
    logging.info("Wrote %d predictions to %s", len(rows), cfg.output_file)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run_model_inference.sh",
        description="Run inference with the 24-D edge TopoQA model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./run_model_inference.sh --config config.yaml.BM55-AF2\n"
            "  ./run_model_inference.sh --config config.yaml --dump-metadata\n"
            "  ./run_model_inference.sh --data-dir data --work-dir work --results-dir results "
            "--checkpoint-path best.ckpt --dataset-name HAF2\n"
            "\n"
            "Direct module invocation:\n"
            "  python -m qtdaqa.new_dynamic_features.model_inference.inference_topoqa_cpu --config config.yaml\n"
        ),
    )
    parser.add_argument("--config", type=str, default=None, help="Path to inference YAML config")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--dump-metadata", action="store_true", help="Print checkpoint feature metadata and exit")
    parser.add_argument("--data-dir", type=str, help="Evaluation dataset directory (required if --config omitted)")
    parser.add_argument("--work-dir", type=str, help="Working directory root (dataset subfolder auto-created)")
    parser.add_argument("--checkpoint-path", type=str, help="Checkpoint (.ckpt/.chkpt) path")
    parser.add_argument("--results-dir", type=str, help="Directory root for inference results")
    parser.add_argument("--label-file", type=str, help="Optional label CSV")
    parser.add_argument("--dataset-name", type=str, help="Dataset identifier used for subdirectories", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--builder-jobs", type=int, default=None, help="Worker count for graph builder stages."
    )
    parser.add_argument(
        "--reuse-existing-graphs",
        action="store_true",
        help="Reuse graphs already present in the work directory when compatible.",
    )
    parser.add_argument(
        "--check-schema",
        action="store_true",
        help="Validate schema compatibility (checkpoint + local builder + existing graphs) and exit without running inference or building graphs.",
    )
    return parser.parse_args(argv)


def _merge_cli_overrides(config: InferenceConfig, args: argparse.Namespace) -> InferenceConfig:
    data_dir = config.data_dir
    if args.data_dir:
        data_dir = _resolve(args.data_dir, Path.cwd())

    checkpoint_path = config.checkpoint_path
    if args.checkpoint_path:
        checkpoint_path = _resolve(args.checkpoint_path, Path.cwd())

    work_base = config.work_dir_base
    if args.work_dir:
        work_base = _resolve(args.work_dir, Path.cwd())

    results_base = config.results_dir_base
    if args.results_dir:
        results_base = _resolve(args.results_dir, Path.cwd())

    dataset_name = args.dataset_name
    if not dataset_name:
        raise SystemExit("--dataset-name is required.")

    def _maybe_strip_dataset(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        if path.name == dataset_name:
            return path.parent
        return path

    if work_base is None:
        work_base = _maybe_strip_dataset(config.work_dir)
    if results_base is None:
        results_base = _maybe_strip_dataset(config.results_dir)

    missing: List[str] = []
    if data_dir is None:
        missing.append("data_dir")
    if work_base is None:
        missing.append("work_dir")
    if results_base is None:
        missing.append("results_dir")
    if checkpoint_path is None:
        missing.append("checkpoint_path")
    if missing:
        raise SystemExit(f"Missing required inference paths (config + CLI): {', '.join(missing)}")

    work_dir = (work_base / dataset_name).resolve()
    results_dir = (results_base / dataset_name).resolve()
    output_file = results_dir / "inference_results.csv"

    builder_cfg = config.builder or BuilderConfig()
    if args.builder_jobs is not None:
        try:
            jobs_override = int(args.builder_jobs)
        except (TypeError, ValueError) as exc:
            raise SystemExit("--builder-jobs must be an integer.") from exc
        if jobs_override <= 0:
            raise SystemExit("--builder-jobs must be greater than zero.")
        builder_cfg = dataclasses.replace(builder_cfg, jobs=jobs_override)

    return dataclasses.replace(
        config,
        data_dir=data_dir,
        work_dir=work_dir,
        checkpoint_path=checkpoint_path,
        results_dir=results_dir,
        output_file=output_file,
        label_file=_resolve(args.label_file, Path.cwd()) if args.label_file else config.label_file,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else config.num_workers,
        builder=builder_cfg,
        reuse_existing_graphs=args.reuse_existing_graphs or config.reuse_existing_graphs,
        dataset_name=dataset_name,
        work_dir_base=work_base,
        results_dir_base=results_base,
        check_schema=bool(args.check_schema) or config.check_schema,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if not args.dataset_name:
        raise SystemExit("--dataset-name is required.")
    if args.config:
        config_path = Path(args.config).resolve()
        cfg = load_config(config_path)
    else:
        required = {
            "data_dir": args.data_dir,
            "work_dir": args.work_dir,
            "checkpoint_path": args.checkpoint_path,
            "results_dir": args.results_dir,
            "dataset_name": args.dataset_name,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing required arguments without --config: {', '.join(missing)}")

        builder_cfg = BuilderConfig(
            jobs=args.builder_jobs if args.builder_jobs is not None else 4,
        )

        dataset_name = args.dataset_name.strip()
        work_base = _resolve(args.work_dir, Path.cwd())
        results_base = _resolve(args.results_dir, Path.cwd())
        work_dir = (work_base / dataset_name).resolve()
        results_dir = (results_base / dataset_name).resolve()
        cfg = InferenceConfig(
            data_dir=_resolve(args.data_dir, Path.cwd()),
            work_dir=work_dir,
            checkpoint_path=_resolve(args.checkpoint_path, Path.cwd()),
            results_dir=results_dir,
            output_file=results_dir / "inference_results.csv",
            label_file=_resolve(args.label_file, Path.cwd()) if args.label_file else None,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            num_workers=args.num_workers if args.num_workers is not None else 0,
            builder=builder_cfg,
            reuse_existing_graphs=args.reuse_existing_graphs,
            use_checkpoint_schema=True,
            interface_schema={},
            topology_schema={},
            node_schema={},
            edge_schema={},
            training_root=_default_training_root(),
            config_name="(CLI parameters)",
            dataset_name=dataset_name,
            work_dir_base=work_base,
            results_dir_base=results_base,
            check_schema=bool(args.check_schema),
        )

    cfg = _merge_cli_overrides(cfg, args)

    checkpoint_meta = extract_feature_metadata(cfg.checkpoint_path)
    if args.dump_metadata:
        print(json.dumps(checkpoint_meta, indent=2))
        return 0

    _guard_schema_overrides(cfg, checkpoint_meta)
    final_schema = resolve_feature_schema(cfg, checkpoint_meta)
    logging.info("Resolved feature schema: %s", json.dumps(final_schema, indent=2))

    logging.info("Config: %s", json.dumps(dataclasses.asdict(cfg), default=str, indent=2))
    schema_dump_path = cfg.work_dir / "feature_metadata.json"
    schema_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with schema_dump_path.open("w", encoding="utf-8") as handle:
        json.dump(final_schema, handle, indent=2)
    logging.info("Feature metadata written to %s", schema_dump_path)

    if cfg.check_schema:
        _run_schema_check(cfg, final_schema)
        return 0

    run_inference(cfg, final_schema, checkpoint_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
