from __future__ import annotations

import pandas as pd

from qtdaqa.new_dynamic_features.graph_builder2.lib.topology_runner import round_topology_frame


def test_round_topology_frame_applies_precision() -> None:
    df = pd.DataFrame({"ID": ["a", "b"], "x": [1.1234567890123, 2.0000000000004], "y": [10.0, 10.0]})
    round_topology_frame(df, 4)
    assert df.loc[0, "x"] == 1.1235
    assert df.loc[1, "x"] == 2.0
    assert df.loc[0, "y"] == 10.0  # unchanged value still round-trips
    assert df.loc[1, "y"] == 10.0
    assert list(df["ID"]) == ["a", "b"]


def test_round_topology_frame_disabled_with_negative_or_none() -> None:
    df = pd.DataFrame({"ID": ["a"], "x": [1.123456789]})
    original = df.copy()
    round_topology_frame(df, None)
    pd.testing.assert_frame_equal(df, original)
    round_topology_frame(df, -1)
    pd.testing.assert_frame_equal(df, original)
