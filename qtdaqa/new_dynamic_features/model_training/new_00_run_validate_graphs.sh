export GRAPH_DIR=topoqa_10A_no_drift_ARM

echo "Generating graph-dir manifest for ${GRAPH_DIR} ..."

# --workers 0 uses all cpu's
time python -m tools.new_validate_graphs --graph-dir ../graph_builder2/output/${GRAPH_DIR}/graph_data --write-manifest --manifest ../graph_builder2/output/${GRAPH_DIR}/graph_data/graph_manifest.json --workers 0

echo "Done with manifest for ${GRAPH_DIR} ..."
