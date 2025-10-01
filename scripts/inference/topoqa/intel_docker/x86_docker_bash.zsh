#!/bin/bash

# Run x86 container with mounted volumes
docker run --rm -it \
  --platform=linux/amd64 \
  -e PATH="/home/app/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  -e PYTHONPATH="/app:/workspace:$PYTHONPATH" \
  -v ../../../../topoqa_train:/app \
  -v ../../../../datasets:/workspace \
  --user app \
  x86_pytorch:latest \
  bash
