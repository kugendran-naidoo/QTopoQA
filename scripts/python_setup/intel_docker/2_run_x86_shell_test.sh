docker run --rm -it \
  --platform=linux/amd64 \
  -e PATH="/home/app/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  -e PYTHONPATH="/app:/workspace:$PYTHONPATH" \
  -v /Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa:/app \
  -v /Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets:/workspace \
  --user app \
  x86_pytorch:latest \
  bash /app/k_test_python_packages.sh

docker run --rm -it \
  --platform=linux/amd64 \
  -e PATH="/home/app/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  x86_pytorch:latest \
  sh -lc 'command -v mkdssp && mkdssp --version || (echo "mkdssp missing"; exit 127)'
