export GRAPH_DIR=MoL_edge_10A

echo "Generating graph-dir manifest for ${GRAPH_DIR} ..."

time python -m tools.validate_graphs --graph-dir ../graph_builder2/output/${GRAPH_DIR}/graph_data --write-manifest --manifest ../graph_builder2/output/${GRAPH_DIR}/graph_data/graph_manifest.json

echo "Done with manifest for ${GRAPH_DIR} ..."
