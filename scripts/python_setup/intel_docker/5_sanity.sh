docker run --rm -it \
  --platform=linux/amd64 \
  -e PATH="/home/app/.local/bin:/usr/local/bin:/usr/bin:/bin" \
  x86_pytorch:latest \
  sh -lc 'command -v mkdssp && mkdssp --version || (echo "mkdssp missing"; exit 127)'
