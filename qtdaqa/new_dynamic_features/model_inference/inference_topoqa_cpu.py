#!/usr/bin/env python3
"""Inference runner for the 24-D edge TopoQA model."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_MODEL_DIR = (SCRIPT_DIR / "../.." / "new_model_training" / "ARM_cpu_only" / "model").resolve()
if str(TRAIN_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_MODEL_DIR))

from gat_5_edge1 import GNN_edge1_edgepooling  # type: ignore  # noqa: E402

BUILDER_SCRIPT = (SCRIPT_DIR.parents[1] / "new_graph_builder" / "graph_builder.py").resolve()
if not BUILDER_SCRIPT.exists():
    raise FileNotFoundError(f"Graph builder script not found at {BUILDER_SCRIPT}")


@dataclasses.dataclass
class InferenceConfig:
    data_dir: Path
    work_dir: Path
    checkpoint_path: Path
    output_file: Path
    label_file: Optional[Path] = None
    batch_size: int = 32
    num_workers: int = 0
    builder: Dict[str, object] = dataclasses.field(default_factory=dict)
    reuse_existing_graphs: bool = False
    use_checkpoint_schema: bool = True
    edge_schema: Dict[str, object] = dataclasses.field(default_factory=dict)
    topology_schema: Dict[str, object] = dataclasses.field(default_factory=dict)


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


def resolve_feature_schema(cfg: InferenceConfig, checkpoint_meta: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    edge_schema: Dict[str, object] = {}
    topo_schema: Dict[str, object] = {}
    if cfg.use_checkpoint_schema:
        edge_schema.update(checkpoint_meta.get("edge_schema") or {})
        topo_schema.update(checkpoint_meta.get("topology_schema") or {})
    edge_schema.update(cfg.edge_schema or {})
    topo_schema.update(cfg.topology_schema or {})
    if not edge_schema:
        raise ValueError("Edge schema unavailable. Provide one in the checkpoint or config.yaml")
    return {"edge_schema": edge_schema, "topology_schema": topo_schema}


def run_graph_builder(cfg: InferenceConfig) -> Path:
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")
    graph_dir = cfg.work_dir / "graph_data"
    builder_work = cfg.work_dir / "builder_work"
    builder_logs = cfg.work_dir / "builder_logs"

    for path in (graph_dir, builder_work, builder_logs):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BUILDER_SCRIPT),
        "--dataset-dir",
        str(cfg.data_dir),
        "--work-dir",
        str(builder_work),
        "--graph-dir",
        str(graph_dir),
        "--log-dir",
        str(builder_logs),
        "--jobs",
        str(cfg.builder.get("jobs", 4)),
    ]
    if cfg.builder.get("topology_dedup_sort"):
        cmd.append("--topology-dedup-sort")
    if not cfg.builder.get("dump_edges", True):
        cmd.append("--no-dump-graph-edges")
    logging.info("Running graph builder: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return graph_dir


def ensure_graph_dir(cfg: InferenceConfig) -> Path:
    graph_dir = cfg.work_dir / "graph_data"
    if cfg.reuse_existing_graphs and graph_dir.exists() and any(graph_dir.glob("*.pt")):
        logging.info("Reusing existing graphs at %s", graph_dir)
        return graph_dir
    return run_graph_builder(cfg)


def load_config(raw_path: Path) -> InferenceConfig:
    with raw_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = InferenceConfig(
        data_dir=_resolve(data["data_dir"], raw_path.parent),
        work_dir=_resolve(data["work_dir"], raw_path.parent),
        checkpoint_path=_resolve(data["checkpoint_path"], raw_path.parent),
        output_file=_resolve(data["output_file"], raw_path.parent),
        label_file=_resolve(data["label_file"], raw_path.parent) if data.get("label_file") else None,
        batch_size=int(data.get("batch_size", 32)),
        num_workers=int(data.get("num_workers", 0)),
        builder=dict(data.get("builder", {})),
        reuse_existing_graphs=bool(data.get("reuse_existing_graphs", False)),
        use_checkpoint_schema=bool(data.get("use_checkpoint_schema", True)),
        edge_schema=dict(data.get("edge_schema", {})),
        topology_schema=dict(data.get("topology_schema", {})),
    )
    return cfg


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

        # Resolve target using composite and fallback keys
        candidate_keys = [model_key, base_name, normalised_name]
        if "/" in model_key:
            prefix, _, suffix = model_key.partition("/")
            candidate_keys.extend(
                [
                    f"{prefix}/{base_name}",
                    f"{prefix}/{normalised_name}",
                ]
            )

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
                    scores[key] = value

        if target_value:
            for key in composite_keys:
                targets[key] = target_value
            for key in base_keys:
                targets.setdefault(key, target_value)

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


def load_model(cfg: InferenceConfig, edge_schema: Dict[str, object]) -> GNN_edge1_edgepooling:
    edge_dim = int(edge_schema.get("dim", 24))
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
    )
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    model.eval()
    return model


def run_inference(cfg: InferenceConfig, final_schema: Dict[str, Dict[str, object]]) -> None:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = ensure_graph_dir(cfg)

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

    model = load_model(cfg, final_schema["edge_schema"])

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
    parser = argparse.ArgumentParser(description="Run inference with the 24-D edge TopoQA model")
    parser.add_argument("--config", type=str, default=None, help="Path to inference YAML config")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--dump-metadata", action="store_true", help="Print checkpoint feature metadata and exit")
    parser.add_argument("--data-dir", type=str, help="Evaluation dataset directory (required if --config omitted)")
    parser.add_argument("--work-dir", type=str, help="Working directory for builder outputs")
    parser.add_argument("--checkpoint-path", type=str, help="Checkpoint (.ckpt/.chkpt) path")
    parser.add_argument("--output-file", type=str, help="CSV path for predictions")
    parser.add_argument("--label-file", type=str, help="Optional label CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--builder-jobs", type=int, default=4)
    parser.add_argument("--builder-topology-dedup-sort", action="store_true")
    parser.add_argument("--no-builder-dump-edges", action="store_true")
    parser.add_argument("--reuse-existing-graphs", action="store_true")
    parser.add_argument("--no-use-checkpoint-schema", action="store_true")
    parser.add_argument("--edge-schema-json", type=str, help="Override edge schema JSON string")
    parser.add_argument("--topology-schema-json", type=str, help="Override topology schema JSON string")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    if args.config:
        config_path = Path(args.config).resolve()
        cfg = load_config(config_path)
    else:
        required = {
            "data_dir": args.data_dir,
            "work_dir": args.work_dir,
            "checkpoint_path": args.checkpoint_path,
            "output_file": args.output_file,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing required arguments without --config: {', '.join(missing)}")

        cfg = InferenceConfig(
            data_dir=_resolve(args.data_dir, Path.cwd()),
            work_dir=_resolve(args.work_dir, Path.cwd()),
            checkpoint_path=_resolve(args.checkpoint_path, Path.cwd()),
            output_file=_resolve(args.output_file, Path.cwd()),
            label_file=_resolve(args.label_file, Path.cwd()) if args.label_file else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            builder={
                "jobs": args.builder_jobs,
                "topology_dedup_sort": args.builder_topology_dedup_sort,
                "dump_edges": not args.no_builder_dump_edges,
            },
            reuse_existing_graphs=args.reuse_existing_graphs,
            use_checkpoint_schema=not args.no_use_checkpoint_schema,
            edge_schema=json.loads(args.edge_schema_json) if args.edge_schema_json else {},
            topology_schema=json.loads(args.topology_schema_json) if args.topology_schema_json else {},
        )

    checkpoint_meta = extract_feature_metadata(cfg.checkpoint_path)
    if args.dump_metadata:
        print(json.dumps(checkpoint_meta, indent=2))
        return 0

    final_schema = resolve_feature_schema(cfg, checkpoint_meta)
    logging.info("Resolved feature schema: %s", json.dumps(final_schema, indent=2))

    logging.info("Config: %s", json.dumps(dataclasses.asdict(cfg), default=str, indent=2))
    schema_dump_path = cfg.work_dir / "feature_metadata.json"
    schema_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with schema_dump_path.open("w", encoding="utf-8") as handle:
        json.dump(final_schema, handle, indent=2)
    logging.info("Feature metadata written to %s", schema_dump_path)

    run_inference(cfg, final_schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
