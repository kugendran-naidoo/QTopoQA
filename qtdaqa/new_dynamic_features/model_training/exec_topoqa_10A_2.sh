# Mac
export TMPDIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/temp

# AWS
# export TMPDIR=/home/app/phd/github/QTopoQA/temp

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export OUTPUT=11d_topoqa_10A_2

# Mac
export GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data

# AWS
# export GRAPH_DIR=/home/app/phd/github/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data

TMPDIR="$TMPDIR" TEMP="$TEMP" TMP="$TMP" \
./run_full_pipeline.sh --manifest manifests/run_exhaustive.yaml | 
tee ${OUTPUT}_run_exhaustive_$(date +%Y%m%d_%H%M%S).log

