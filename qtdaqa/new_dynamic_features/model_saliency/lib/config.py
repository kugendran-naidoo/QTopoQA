"""
Centralised configuration models for the saliency toolkit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass
class CheckpointConfig:
    checkpoint_path: Path
    repo_root: Path
    device: str = "cpu"
    strict: bool = False

    def validate(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Repository root not found: {self.repo_root}")


@dataclass
class GraphSelection:
    graph_dir: Path
    models: Sequence[str] = field(default_factory=list)
    limit: Optional[int] = None

    def validate(self) -> None:
        if not self.graph_dir.exists():
            raise FileNotFoundError(f"Graph directory not found: {self.graph_dir}")

    def iter_models(self) -> Iterable[str]:
        if self.models:
            records = list(self.models)
        else:
            records = [path.stem for path in sorted(self.graph_dir.glob("*.pt"))]
        if self.limit and self.limit > 0:
            records = records[: self.limit]
        yield from records


@dataclass
class SaliencyRequest:
    checkpoint: CheckpointConfig
    graphs: GraphSelection
    output_dir: Path
    graph_metadata_path: Optional[Path] = None
    graph_summary_path: Optional[Path] = None
    use_half_precision: bool = False
    gradient_method: str = "integrated_gradients"
    integration_steps: int = 64
    noise_tunnel_samples: int = 0
    edge_gradient_method: Optional[str] = None
    pyg_explainer: bool = True
    randomization_trials: int = 0

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def default_output_dir(run_name: str, root: Optional[Path] = None) -> Path:
    base = Path(root) if root else Path("outputs")
    return (base / run_name).resolve()
