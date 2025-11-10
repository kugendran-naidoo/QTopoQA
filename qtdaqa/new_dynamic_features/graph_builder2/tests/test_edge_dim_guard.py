import numpy as np
import pytest

from qtdaqa.new_dynamic_features.graph_builder2.lib.edge_dim_guard import (
    determine_edge_feature_dim,
    ensure_edge_feature_dim,
)


def test_determine_edge_feature_dim_handles_zero_rows():
    arr = np.empty((0, 24), dtype=np.float32)
    assert determine_edge_feature_dim(arr) == 24


def test_ensure_edge_feature_dim_infers_when_metadata_missing():
    arr = np.zeros((3, 5), dtype=np.float32)
    assert ensure_edge_feature_dim("edge/test_module", arr, None) == 5


def test_ensure_edge_feature_dim_detects_mismatch():
    arr = np.zeros((2, 11), dtype=np.float32)
    with pytest.raises(ValueError):
        ensure_edge_feature_dim("edge/test_module", arr, 24)


def test_ensure_edge_feature_dim_rejects_zero_width():
    arr = np.empty((0, 0), dtype=np.float32)
    with pytest.raises(ValueError):
        ensure_edge_feature_dim("edge/test_module", arr, None)
