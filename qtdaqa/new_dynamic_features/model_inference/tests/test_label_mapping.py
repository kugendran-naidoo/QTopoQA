from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

def _discover_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


REPO_ROOT = _discover_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.model_inference.inference_topoqa_cpu import (  # noqa: E402
    GraphInferenceDataset,
    load_label_map,
)


def _write_stub_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = Data(
        x=torch.zeros((1, 1)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, 1)),
    )
    torch.save(data, path)


def test_target_specific_labels_preferred(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Decoys that share the same base name across targets must read the correct DockQ."""

    label_csv = tmp_path / "labels.csv"
    label_csv.write_text(
        "Target,Model,DockQ\n"
        "3SE8,ranked_0,0.720\n"
        "4M5Z,ranked_0,0.111\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING)
    label_info = load_label_map(label_csv)

    graph_items = []
    for target in ("3SE8", "4M5Z"):
        graph_path = tmp_path / "graphs" / target / "ranked_0_tidy.pt"
        _write_stub_graph(graph_path)
        graph_items.append((f"{target}/ranked_0_tidy", graph_path))

    dataset = GraphInferenceDataset(graph_items, label_info.scores, label_info.targets)

    sample_a = dataset[0]
    sample_b = dataset[1]

    assert str(sample_a.target) == "3SE8"
    assert pytest.approx(float(sample_a.y)) == 0.720
    assert str(sample_b.target) == "4M5Z"
    assert pytest.approx(float(sample_b.y)) == 0.111

    assert any("duplicate unqualified model names" in record.message for record in caplog.records)
