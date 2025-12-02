import os
import sys
from pathlib import Path

# Ensure repo root (QTopoQA) is on sys.path so `qtdaqa` imports resolve.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QTOPO_SKIP_MODULE_REGISTRY", "1")
os.environ.setdefault("QTOPO_ALLOW_MODULE_OVERRIDE", "1")

from qtdaqa.new_dynamic_features.graph_builder2.tests.utils_stub_modules import *  # noqa: F401,F403
