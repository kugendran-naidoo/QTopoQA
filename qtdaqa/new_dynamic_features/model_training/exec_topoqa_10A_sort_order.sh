# Mac
export TMPDIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/temp

# AWS
# export TMPDIR=/home/app/phd/github/QTopoQA/temp

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export OUTPUT=11d_topoqa_10A_sort_order

# Mac
export GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data

# AWS
# export GRAPH_DIR=/home/app/phd/github/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data

# Generate this after graph_builder2

# FIRST use of graph sort order validation
# write the manifest - maybe this must move to graph_builder2
# python -m tools.validate_graphs \
#   --graph-dir ${GRAPH_DIR} \
#   --write-manifest \
#   --manifest ${GRAPH_DIR}/graph_manifest.json

# Check before running model_training

# FIRST use of graph sort order validation
# check the manifest
# python -m tools.validate_graphs --ignore-metadata \
#   --graph-dir ${GRAPH_DIR} \
#   --manifest ${GRAPH_DIR}/graph_manifest.json

# DO NOT USE UNLESS YOUR GRAPHS ARE NOT SORTED - EMERGENCY feature
# export CANONICALIZE_GRAPHS_ON_LOAD=1

TMPDIR="$TMPDIR" TEMP="$TEMP" TMP="$TMP" \
VALIDATE_GRAPHS=1 \
VALIDATE_GRAPHS_MANIFEST=${GRAPH_DIR}/graph_manifest.json \
./run_full_pipeline.sh --manifest manifests/run_exhaustive.yaml | 
tee ${OUTPUT}_run_exhaustive_$(date +%Y%m%d_%H%M%S).log

