from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qtdaqa.new_dynamic_features.graph_builder.lib.node_utils import canonical_id_order


def test_canonical_id_order_returns_sorted_indices() -> None:
    ids = ["c<C>", "c<A>", "c<B>"]

    order = canonical_id_order(ids)

    assert order == [1, 2, 0]
