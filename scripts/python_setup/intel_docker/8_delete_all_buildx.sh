#!/usr/bin/env sh
# delete_all_buildx_builders.sh
# POSIX-compliant script to remove all docker buildx servers/builders

set -eu

echo "==> Listing all buildx builders..."
builders=$(docker buildx ls --format '{{.Name}}' 2>/dev/null || true)

if [ -z "$builders" ]; then
    echo "No buildx builders found."
    exit 0
fi

for b in $builders; do
    echo "Removing builder: $b"
    docker buildx rm -f "$b" >/dev/null 2>&1 || {
        echo "Warning: Failed to remove builder $b"
    }
done

echo "==> Cleanup complete."

