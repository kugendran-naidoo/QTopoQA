"""
Wrapper around torch_geometric.explain Explainer utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.explain import Explainer, GNNExplainer

from pytorch_lightning import LightningModule
from torch_geometric.data import Data


@dataclass
class ExplainerConfig:
    epochs: int = 200
    lr: float = 0.01
    silent: bool = True


def run_gnn_explainer(
    model: LightningModule,
    data: Data,
    *,
    config: ExplainerConfig | None = None,
) -> Optional[dict]:
    if config is None:
        config = ExplainerConfig()
    if data.edge_index is None:
        return None
    model.eval()
    data = data.clone()
    explain = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=config.epochs, lr=config.lr),
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="continuous",
        ),
    )
    with torch.no_grad():
        return explain(data.x, data.edge_index, edge_attr=data.edge_attr, batch=getattr(data, "batch", None))
