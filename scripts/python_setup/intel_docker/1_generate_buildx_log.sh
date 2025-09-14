docker buildx build --load --progress=plain -t x86_pytorch:latest . 2>&1 | tee build.log
