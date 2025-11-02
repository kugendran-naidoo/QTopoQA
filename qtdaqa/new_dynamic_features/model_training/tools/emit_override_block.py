#!/usr/bin/env python3
"""
Convenience entrypoint to emit commented override blocks for graph feature metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qtdaqa.new_dynamic_features.common.feature_inspector import main as inspector_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Ensure the override block flag is always forwarded for this helper.
    if "--emit-override-block" not in args:
        args.append("--emit-override-block")
    return inspector_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
