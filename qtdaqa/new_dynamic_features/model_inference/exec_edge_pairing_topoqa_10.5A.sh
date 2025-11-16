# picks the best model from inference automatically or manually if specified in config yaml

export WORK_DIR=edge_pairing_topoqa_10.5A

export DATASET_1=BM55-AF2
export DATASET_2=HAF2
export DATASET_3=ABAG-AF3

for DATASET in "$DATASET_1" "$DATASET_2" "$DATASET_3"; do

    echo "Running Inference on $DATASET in ${WORK_DIR}"

time ./run_model_inference.sh \
--dataset-name ${DATASET} \
--config qtdaqa/new_dynamic_features/model_inference/config.yaml.${DATASET} \
--work-dir qtdaqa/new_dynamic_features/model_inference/output/${WORK_DIR}/work \
--results-dir qtdaqa/new_dynamic_features/model_inference/output/${WORK_DIR}/results \
--log-level INFO 2>&1 | tee ${DATASET}_${WORK_DIR}_$(date +%Y%m%d_%H%M%S).log &

done

wait

for DATASET in "$DATASET_1" "$DATASET_2" "$DATASET_3"; do

   ./run_results_summary.sh --results-dir output/${WORK_DIR}/results/${DATASET} \

done
