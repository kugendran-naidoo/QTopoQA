from pathlib import Path

from qtdaqa.new_dynamic_features.model_training.common.graph_cache import GraphTensorCache


def test_graph_tensor_cache_hits_once():
    cache = GraphTensorCache(max_items=4)
    path = Path("/tmp/sample.pt")
    calls = {"count": 0}

    def loader(p: Path):
        calls["count"] += 1
        return f"payload:{p}"

    value1 = cache.get(path, loader)
    value2 = cache.get(path, loader)

    assert value1 == value2
    assert calls["count"] == 1


def test_graph_tensor_cache_eviction():
    cache = GraphTensorCache(max_items=2)
    paths = [Path(f"/tmp/{idx}.pt") for idx in range(3)]
    outputs = []

    def loader(p: Path):
        outputs.append(str(p))
        return f"obj:{p}"

    for p in paths:
        cache.get(p, loader)

    # Oldest entry (0.pt) should have been evicted.
    cache.get(paths[0], loader)
    assert outputs.count(str(paths[0])) == 2
