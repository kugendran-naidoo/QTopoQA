from __future__ import annotations

from typing import List, Sequence


def canonical_id_order(ids: Sequence[str]) -> List[int]:
    order = list(range(len(ids)))
    order.sort(key=lambda idx: ids[idx])
    return order
