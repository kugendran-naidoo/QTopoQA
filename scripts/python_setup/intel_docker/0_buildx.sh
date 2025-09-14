#!/bin/sh
# buildx.sh — POSIX-compliant Docker Buildx helper
# Works with /bin/sh on macOS (zsh, bash) and Linux shells.

# Defaults (overridden by flags or env vars)
: "${IMAGE_NAME:=x86_pytorch}"
: "${DOCKERFILE:=Dockerfile}"
: "${PLATFORM:=linux/amd64}"
: "${BUILDX_SERVER:=buildx_server}"
: "${PUSH:=local}"    # allowed: local | registry | tar:<path>

set -eu

# ------------- helpers -------------
log() { printf '%s\n' "==> $*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<'USAGE'
Usage: buildx.sh [options]

Options:
  -i IMAGE_NAME     Image name (default: x86_pytorch)
  -f DOCKERFILE     Dockerfile path (default: Dockerfile)
  -P PLATFORM       Target platform(s), e.g. linux/amd64 or linux/amd64,linux/arm64 (default: linux/amd64)
  -b BUILDX_SERVER  Buildx builder name (default: buildx_server)
  -x PUSH_MODE      Push mode: "local", "registry", or "tar:/path/to/image.tar" (default: local)
  -t TAG            Image tag (default: "latest")
  -c CONTEXT        Build context (default: .)
  -h                Show help

You can also set the corresponding environment variables:
  IMAGE_NAME, DOCKERFILE, PLATFORM, BUILDX_SERVER, PUSH

Notes:
  • PUSH=local  -> uses --load (single-platform only).
  • PUSH=registry -> uses --push (multi-platform OK).
  • PUSH=tar:/path/file.tar -> writes OCI/Docker archive to path (multi-platform OK).
USAGE
}

# ------------- parse args -------------
TAG="latest"
CONTEXT="."

# POSIX getopts
while getopts "i:f:P:b:x:t:c:h" opt; do
  case "$opt" in
    i) IMAGE_NAME=$OPTARG ;;
    f) DOCKERFILE=$OPTARG ;;
    P) PLATFORM=$OPTARG ;;
    b) BUILDX_SERVER=$OPTARG ;;
    x) PUSH=$OPTARG ;;
    t) TAG=$OPTARG ;;
    c) CONTEXT=$OPTARG ;;
    h) usage; exit 0 ;;
    \?) usage; exit 2 ;;
  esac
done

# ------------- validations -------------
have docker || die "Docker is not installed or not on PATH."
have docker || exit 1
docker version >/dev/null 2>&1 || die "Docker daemon not reachable."
docker buildx version >/dev/null 2>&1 || die "Docker Buildx is not available."

[ -f "$DOCKERFILE" ] || die "Dockerfile not found at: $DOCKERFILE"
[ -d "$CONTEXT" ] || die "Build context not found: $CONTEXT"

case "$PUSH" in
  local) : ;;
  registry) : ;;
  tar:*) : ;;
  *) die "Invalid PUSH mode: $PUSH (use 'local', 'registry', or 'tar:/path/file.tar')" ;;
esac

# local mode cannot handle multiple platforms via --load
case "$PUSH" in
  local)
    case "$PLATFORM" in
      *,*) die "PUSH=local only supports a single platform; got '$PLATFORM'. Use PUSH=registry or PUSH=tar:/path/file.tar." ;;
      *) : ;;
    esac
    ;;
esac

# ------------- idempotent builder reset -------------
log "Resetting Buildx builder: $BUILDX_SERVER"

# Remove existing builder if present
if docker buildx inspect "$BUILDX_SERVER" >/dev/null 2>&1; then
  log "Removing existing builder: $BUILDX_SERVER"
  docker buildx rm -f "$BUILDX_SERVER" >/dev/null 2>&1 || true
fi

# Kill stray buildkit containers associated with this builder
# Common container names: buildx_buildkit_<BUILDERNAME>, buildx_buildkit_<BUILDERNAME>0, etc.
# We remove any container whose name contains "buildx_buildkit_${BUILDX_SERVER}"
STRAYS=$(docker ps -aq --filter "name=buildx_buildkit_${BUILDX_SERVER}" || true)
if [ -n "${STRAYS:-}" ]; then
  log "Stopping stray buildkit containers: $STRAYS"
  # Stop then remove; ignore errors
  docker stop $STRAYS >/dev/null 2>&1 || true
  docker rm -f $STRAYS >/dev/null 2>&1 || true
fi

# Create & use fresh builder
log "Creating fresh builder: $BUILDX_SERVER"
docker buildx create --name "$BUILDX_SERVER" --driver docker-container --use >/dev/null

# Boot the builder (ensures it’s ready)
log "Bootstrapping builder"
docker buildx inspect --bootstrap >/dev/null

# ------------- compute outputs -------------
IMAGE_REF="${IMAGE_NAME}:${TAG}"

BUILD_ARGS="--file \"$DOCKERFILE\" --platform \"$PLATFORM\" -t \"$IMAGE_REF\""

# Decide output/push flags
OUTPUT_FLAG=""
case "$PUSH" in
  local)
    # Single-platform load into local docker
    OUTPUT_FLAG="--load"
    ;;
  registry)
    OUTPUT_FLAG="--push"
    ;;
  tar:*)
    OUT_PATH=${PUSH#tar:}
    # Normalize directory and ensure it exists
    OUT_DIR=$(dirname -- "$OUT_PATH")
    [ -d "$OUT_DIR" ] || mkdir -p "$OUT_DIR"
    # Use OCI layout (compatible) tarball
    OUTPUT_FLAG="--output=type=oci,dest=\"$OUT_PATH\""
    ;;
esac

# ------------- build -------------
log "Building image"
log "  Image     : $IMAGE_REF"
log "  Dockerfile: $DOCKERFILE"
log "  Platform  : $PLATFORM"
log "  Builder   : $BUILDX_SERVER"
log "  Push mode : $PUSH"
log "  Context   : $CONTEXT"

# shellcheck disable=SC2086 # (POSIX: we’re guarding with quotes in BUILD_ARGS content)
# We must eval because BUILD_ARGS/OUTPUT_FLAG contain quoted segments
CMD="docker buildx build $OUTPUT_FLAG $BUILD_ARGS \"$CONTEXT\""
# For transparency:
log "Running: $CMD"
# Execute:
# shellcheck disable=SC3045
# (SC3045 is non-POSIX in some linters; this is pure POSIX eval)
eval "$CMD"

log "Done."

