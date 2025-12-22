from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Any, Dict

class GraphTensorCache:
    def __init__(self, max_items: int = 128):
        self.max_items = max(1, int(max_items))
        self._lock = threading.Lock()
        self._order: "OrderedDict[str, Any]" = OrderedDict()

    def __getstate__(self) -> Dict[str, Any]:
        # Avoid pickling the lock or cached tensors when DataLoader uses spawn.
        return {"max_items": self.max_items}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.max_items = max(1, int(state.get("max_items", 128)))
        self._lock = threading.Lock()
        self._order = OrderedDict()

    def get(self, path: Path, loader: Callable[[Path], Any]) -> Any:
        key = str(path)
        with self._lock:
            if key in self._order:
                value = self._order.pop(key)
                self._order[key] = value
                return value
        data = loader(path)
        with self._lock:
            self._order[key] = data
            if len(self._order) > self.max_items:
                self._order.popitem(last=False)
        return data
