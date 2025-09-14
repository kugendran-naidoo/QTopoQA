#!/usr/bin/env bash
set -euo pipefail

# ---------- Config (override via env) ----------
IMAGE_NAME="${IMAGE_NAME:-app-p3108}"
IMAGE_TAG="${IMAGE_TAG:-v0.1.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.8}"   # used only if your Dockerfile accepts this ARG

# Build mode: load -> load into local Docker; push -> push to ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
BUILD_MODE="${BUILD_MODE:-load}"              # load | push
REGISTRY="${REGISTRY:-}"                      # e.g. docker.io/you or ghcr.io/org (empty = local-only)

# Builder settings
BUILDER_NAME="${BUILDER_NAME:-container-builder}"
BUILDER_DRIVER="${BUILDER_DRIVER:-docker-container}"
ENSURE_BINFMT="${ENSURE_BINFMT:-1}"
RESET_BUILDER="${RESET_BUILDER:-0}"
# ------------------------------------------------

final_ref="${IMAGE_NAME}:${IMAGE_TAG}"
if [[ -n "${REGISTRY}" ]]; then
  final_ref="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "==> Image        : ${final_ref}"
echo "==> Build mode   : ${BUILD_MODE}"
echo "==> Python pin   : ${PYTHON_VERSION}"
echo "==> Builder name : ${BUILDER_NAME}"

# 0) Sanity checks
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 127; }
docker buildx version >/dev/null 2>&1 || { echo "ERROR: docker buildx plugin not available"; exit 127; }

# 1) Ensure we are NOT on Colima for this workflow
active_ctx="$(docker context show || echo "")"
echo "==> Active Docker context: ${active_ctx}"
if [[ "${active_ctx}" == "colima" ]]; then
  echo "==> Switching from 'colima' to Docker Desktop context..."
  (docker context use default || docker context use desktop-linux)
  echo "==> Now using context: $(docker context show)"
fi

# 2) Idempotent builder setup
if [[ "${RESET_BUILDER}" == "1" ]]; then
  echo "==> Removing existing builder (if any): ${BUILDER_NAME}"
  docker buildx rm -f "${BUILDER_NAME}" >/dev/null 2>&1 || true
fi

if docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
  echo "==> Using existing docker buildx builder: ${BUILDER_NAME}"
  docker buildx use "${BUILDER_NAME}" >/dev/null
else
  echo "==> Creating docker buildx builder (${BUILDER_DRIVER}): ${BUILDER_NAME}"
  docker buildx create --name "${BUILDER_NAME}" --driver "${BUILDER_DRIVER}" --use >/dev/null
fi

# Bootstrap builder & (optionally) ensure binfmt on Docker Desktop
docker buildx inspect --bootstrap >/dev/null || true
if [[ "${ENSURE_BINFMT}" == "1" ]] && docker info 2>/dev/null | grep -qi desktop; then
  echo "==> Ensuring binfmt for amd64 (Docker Desktop detected)"
  docker run --privileged --rm tonistiigi/binfmt --install amd64 >/dev/null 2>&1 || true
fi

# 3) Build (linux/amd64) and load/push
build_args=(
  --platform linux/amd64
  -t "${final_ref}"
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
  -f Dockerfile
  .
)

if [[ "${BUILD_MODE}" == "load" ]]; then
  echo "==> Building (and loading) linux/amd64 image locally..."
  docker buildx build "${build_args[@]}" --load
elif [[ "${BUILD_MODE}" == "push" ]]; then
  echo "==> Building (and pushing) linux/amd64 image..."
  docker buildx build "${build_args[@]}" --push
else
  echo "ERROR: BUILD_MODE must be 'load' or 'push'." >&2
  exit 1
fi

# 4) Verify inside the image (explicit platform when running on Apple Silicon)
echo "==> Verifying Python inside the image..."
docker run --rm --platform=linux/amd64 "${final_ref}" python - <<'PY'
import platform
print("Python", platform.python_version())
PY

echo "==> Done."

