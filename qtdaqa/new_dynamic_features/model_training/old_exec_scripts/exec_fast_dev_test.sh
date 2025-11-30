#!/usr/bin/env bash
# Fast dev run for sanity checks (small manifest)

# Mac
export TMPDIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/temp

# AWS
# export TMPDIR=/home/app/phd/github/QTopoQA/temp

export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# Use your fast-dev manifest
MANIFEST=manifests/run_fast_dev.yaml

export OUTPUT=edge_plus_min_agg_topo_10A
export export CHECK=edge_plus_min_agg_topo_10A

export GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${OUTPUT}/graph_data
export CHECK_GRAPH_DIR=/Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/qtdaqa/new_dynamic_features/graph_builder2/output/${CHECK}/graph_data

export VALIDATE_GRAPHS=1
export VALIDATE_GRAPHS_IGNORE_METADATA=1
export VALIDATE_GRAPHS_MANIFEST=${CHECK_GRAPH_DIR}/graph_manifest.json

TMPDIR="$TMPDIR" TEMP="$TEMP" TMP="$TMP" \
./run_full_pipeline.sh --manifest "$MANIFEST" \
  | tee fast_dev_$(date +%Y%m%d_%H%M%S).log

