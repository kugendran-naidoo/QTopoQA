echo "Sanity check: listing input dir in container"
docker run --rm \
  --platform=linux/amd64 \
  -v "/Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets:/workspace" \
  x86_pytorch:latest \
  sh -lc 'ls -l /workspace/BM55-AF2/decoy/3SE8 | head; \
          echo "Sample file first lines:"; \
          f=$(ls /workspace/BM55-AF2/decoy/3SE8 | head -n1); \
          head -n 3 "/workspace/BM55-AF2/decoy/3SE8/$f" || true'
