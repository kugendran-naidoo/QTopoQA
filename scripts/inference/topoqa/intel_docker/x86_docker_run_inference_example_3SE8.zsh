#!/bin/bash

# Run x86 container with mounted volumes
docker run --rm -it \
  --platform=linux/amd64 \
  -e PATH="/home/app/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  -e PYTHONPATH="/app:/workspace:$PYTHONPATH" \
  -v ../../../../topoqa:/app \
  -v ../../../../datasets/examples:/workspace \
  -v ./logs:/logs \
  --user app \
  x86_pytorch:latest \
  python /app/k_mac_inference_pca_tsne4.py \
    --complex-folder \
    /workspace/BM55-AF2/decoy/3SE8 \
    --work-dir \
    /logs/output/work \
    --results-dir \
    /logs/output/results \
    --checkpoint \
    /app/model/topoqa.ckpt \
    --jobs 8 \
    --num-workers 2 \
    --batch-size 4 \
    --device cpu \
    --cutoff 10 \
    --overwrite

