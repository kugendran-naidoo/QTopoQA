OUTPUT="output/smoke_pilot_test"
FEATURE_FILE="feature_configs/feature-config.yaml.topoqa.10A"

# mk dirs
mkdir -p ${OUTPUT}/work
mkdir -p ${OUTPUT}/graph_data

time ./run_graph_builder2.sh --pdb-warnings \
--dataset-dir datasets/training/adjusted/pilot_batch_Dockground_MAF2 \
--work-dir qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/work \
--graph-dir qtdaqa/new_dynamic_features/graph_builder2/${OUTPUT}/graph_data \
--log-dir qtdaqa/new_dynamic_features/graph_builder2/logs_${OUTPUT} \
--jobs 16 \
--feature-config qtdaqa/new_dynamic_features/graph_builder2/${FEATURE_FILE}
