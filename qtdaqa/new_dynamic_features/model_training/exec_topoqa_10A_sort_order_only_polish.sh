# Mac
export TMPDIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/temp

# AWS
# export TMPDIR=/home/app/phd/github/QTopoQA/temp

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

export OUTPUT=11d_topoqa_10A_sort_order

# Mac
export GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data

export VALIDATE_GRAPHS=1
export VALIDATE_GRAPHS_MANIFEST=${GRAPH_DIR}/graph_manifest.json
export PYTHONPATH=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA

LAUNCH=1 MANIFEST_PATH=manifests/run_polish_heavy.yaml \
./polish_from_best.sh |
tee ${OUTPUT}_polish_from_best_$(date +%Y%m%d_%H%M%S).log

