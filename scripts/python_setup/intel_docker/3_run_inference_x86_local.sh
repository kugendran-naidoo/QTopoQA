echo
printf "Run inference test from ../../inference/topoqa/intel_docker/\n"
printf "Run example via x86_docker_run_inference_example_3SE8.zsh\n"

##!/bin/bash
#
## Run x86 container with mounted volumes
#docker run --rm -it \
#  --platform=linux/amd64 \
#  -e PATH="/home/app/.local/bin:/usr/local/bin:/usr/bin:/bin" \
#  -e PYTHONPATH="/app:/workspace:$PYTHONPATH" \
#  -v /Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa:/app \
#  -v /Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets:/workspace \
#  --user app \
#  x86_pytorch:latest \
#  python /app/k_mac_inference_pca_tsne4.py \
#    --complex-folder \
#    /workspace/BM55-AF2/decoy/3SE8 \
#    --work-dir \
#    /workspace/delete_temp_output/work \
#    --results-dir \
#    /workspace/delete_temp_output/results \
#    --checkpoint \
#    /app/model/topoqa.ckpt \
#    --jobs 8 \
#    --num-workers 2 \
#    --batch-size 4 \
#    --device cpu \
#    --cutoff 10 \
#    --overwrite
#
