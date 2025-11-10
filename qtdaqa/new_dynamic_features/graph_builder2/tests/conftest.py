import os

os.environ.setdefault("QTOPO_SKIP_MODULE_REGISTRY", "1")
os.environ.setdefault("QTOPO_ALLOW_MODULE_OVERRIDE", "1")

from qtdaqa.new_dynamic_features.graph_builder2.tests.utils_stub_modules import *  # noqa: F401,F403
