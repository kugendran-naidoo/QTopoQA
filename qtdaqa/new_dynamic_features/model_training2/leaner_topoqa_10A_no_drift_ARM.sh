# General
export OUTPUT=topoqa_10A_no_drift_ARM
export CHECK=topoqa_10A_no_drift_ARM

# Mac
export TMPDIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/temp
export GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data
export CHECK_GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${CHECK}/graph_data
export PYTHONPATH=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA

# AWS
# export TMPDIR=/home/app/phd/github/QTopoQA/temp
# export GRAPH_DIR=/home/app/phd/github/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data
# export CHECK_GRAPH_DIR=/home/app/phd/github/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${CHECK}/graph_data
# export PYTHONPATH=/home/app/phd/github/QTopoQA

# Generate this after graph_builder2 for a golden hash

# FIRST use of graph sort order validation
# write the manifest - maybe this must move to graph_builder2
# python -m tools.validate_graphs \
#   --graph-dir ${GRAPH_DIR} \
#   --write-manifest \
#   --manifest ${GRAPH_DIR}/graph_manifest.json

# Check before running model_training

echo "Skipped Validating graphs using standalone tool ..."

# FIRST use of graph sort order validation
# check the manifest
# python -m tools.validate_graphs --ignore-metadata \
# --graph-dir ${GRAPH_DIR} \
# --manifest ${CHECK_GRAPH_DIR}/graph_manifest.json

# DO NOT USE UNLESS YOUR GRAPHS ARE NOT SORTED - EMERGENCY feature
# export CANONICALIZE_GRAPHS_ON_LOAD=1

export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export VALIDATE_GRAPHS=1
export VALIDATE_GRAPHS_IGNORE_METADATA=1
export VALIDATE_GRAPHS_MANIFEST=${CHECK_GRAPH_DIR}/graph_manifest.json

echo "Starting model training ..."

# run initial model training
./run_full_pipeline.sh --manifest manifests/run_lean.yaml | 
tee ${OUTPUT}_run_lean_$(date +%Y%m%d_%H%M%S).log

echo "Starting post model training polishing ..."

# run post training polish from best
LAUNCH=1 MANIFEST_PATH=manifests/run_polish_lean.yaml \
./polish_from_best.sh |
tee ${OUTPUT}_lean_polish_from_best_$(date +%Y%m%d_%H%M%S).log

