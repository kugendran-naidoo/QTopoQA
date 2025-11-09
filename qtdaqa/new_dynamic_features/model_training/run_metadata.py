from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional


def update_run_metadata(save_dir: Path, mutator: Callable[[Dict[str, object]], None]) -> None:
    metadata_path = save_dir / "run_metadata.json"
    metadata: Dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    mutator(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def record_selection_metadata(
    save_dir: Path,
    *,
    primary_metric: str,
    use_val_spearman: bool,
    spearman_weight: float,
    spearman_min_delta: float,
) -> None:
    def _mutate(metadata: Dict[str, object]) -> None:
        metadata["selection_primary_metric"] = primary_metric
        selection_section = metadata.setdefault("selection", {})
        selection_section["primary_metric"] = primary_metric
        selection_section["use_val_spearman"] = use_val_spearman
        selection_section["spearman_weight"] = spearman_weight
        selection_section["spearman_min_delta"] = spearman_min_delta
        selection_section["secondary_metric"] = "val_spearman_corr" if use_val_spearman else None

    update_run_metadata(save_dir, _mutate)


def record_checkpoint_paths(
    save_dir: Path,
    *,
    primary_metric: str,
    primary_path: Optional[Path],
    alternates: Optional[Dict[str, Optional[Path]]] = None,
) -> None:
    def _mutate(metadata: Dict[str, object]) -> None:
        checkpoints = metadata.setdefault("checkpoints", {})
        checkpoints["primary_metric"] = primary_metric
        checkpoints["primary_path"] = str(primary_path) if primary_path else None
        if alternates:
            serialised: Dict[str, Optional[str]] = {}
            for metric_name, path in alternates.items():
                serialised[str(metric_name)] = str(path) if path else None
            checkpoints["alternates"] = serialised

    update_run_metadata(save_dir, _mutate)


def resolve_checkpoint_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.exists():
        try:
            return candidate.resolve()
        except OSError:
            return candidate
    if candidate.suffix == ".ckpt":
        alt = candidate.with_suffix(".chkpt")
        if alt.exists():
            try:
                return alt.resolve()
            except OSError:
                return alt
    return candidate
