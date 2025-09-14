docker run --rm -it \
  -v /Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/topoqa:/app \
  -v /Volumes/PData/Data/Dev/Github/Repos/phd3/topoqa/datasets:/workspace \
  --user app \
  x86_pytorch:latest \
  bash /app/k_test_python_packages.sh

