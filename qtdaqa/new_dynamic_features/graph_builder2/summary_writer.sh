PYTHONPATH=$(pwd) python - <<'PY'
from pathlib import Path
from qtdaqa.new_dynamic_features.graph_builder2.lib.schema_summary import write_schema_summary

graph_dir = Path("qtdaqa/new_dynamic_features/graph_builder2/output/edge_plus_min_agg_topo_heavy_10A/graph_data")
print("Writing summary for", graph_dir)
print(write_schema_summary(graph_dir))
PY
