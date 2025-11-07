"""
Prepared DataLoader helpers for saliency analysis (single graphs or batches).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch_geometric.data import Data

from .metadata import load_graph, resolve_graph_paths


@dataclass
class GraphSample:
    name: str
    data: Data


def load_graph_samples(graph_dir: Path, models: Sequence[str]) -> List[GraphSample]:
    samples: List[GraphSample] = []
    for name, path in resolve_graph_paths(graph_dir, models):
        data = load_graph(path)
        samples.append(GraphSample(name=name, data=data))
    return samples


def ensure_batch(data: Data) -> Data:
    if getattr(data, "batch", None) is None:
        n_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0)
        data.batch = torch.zeros(n_nodes, dtype=torch.long)
    return data
