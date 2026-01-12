from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from qtdaqa.new_dynamic_features.model_training2.common.feature_mask import load_feature_mask
from qtdaqa.new_dynamic_features.model_training2.model_train_topoqa_cpu import GraphRegressionDataset


def test_load_feature_mask_json_list(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps([1, 0, 1]), encoding="utf-8")
    mask = load_feature_mask(mask_path)
    assert mask.tolist() == [1.0, 0.0, 1.0]


def test_graph_regression_dataset_applies_node_mask(tmp_path: Path) -> None:
    data = Data(x=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    graph_path = tmp_path / "sample.pt"
    torch.save(data, graph_path)

    samples = [("sample", graph_path, torch.tensor([1.0]))]
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    dataset = GraphRegressionDataset(
        samples,
        feature_mask=mask,
        feature_mask_apply_to="node",
        feature_mask_strict=True,
    )
    loaded = dataset[0]
    assert loaded.x.tolist() == [[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]]
