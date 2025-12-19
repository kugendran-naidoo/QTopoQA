#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====
DSSP_TAG="v4.5.7"                     # mkdssp git tag to build
SRC_ROOT="$HOME/src"                  # where we'll put source trees
FMT_PREFIX="$HOME/opt/fmt"            # where fmt will be installed
DSSP_PREFIX="$HOME/opt/mkdssp-4.5.7"  # where mkdssp will be installed
CMAKE="${CMAKE:-cmake}"               # allow override via env if you want

echo "Using CMake:"
"$CMAKE" --version

# Recommended: Apple Clang on macOS is fine; if you want GCC:
# export CC="$(brew --prefix)/bin/gcc-14"
# export CXX="$(brew --prefix)/bin/g++-14"

mkdir -p "$SRC_ROOT"
mkdir -p "$(dirname "$FMT_PREFIX")"
mkdir -p "$(dirname "$DSSP_PREFIX")"

# ===== Step 1: Build fmt from source (if not already) =====
if [ ! -d "$FMT_PREFIX/lib" ] && [ ! -d "$FMT_PREFIX/lib64" ]; then
  echo ">>> Building fmt into $FMT_PREFIX"
  cd "$SRC_ROOT"

  if [ ! -d fmt ]; then
    git clone https://github.com/fmtlib/fmt.git
  fi

  cd fmt
  # Pin a known good version if you like, otherwise use latest:
  # git checkout 10.2.0

  rm -rf build
  "$CMAKE" -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX="$FMT_PREFIX"

  "$CMAKE" --build build -j"$(sysctl -n hw.ncpu)"
  "$CMAKE" --install build
else
  echo ">>> Reusing existing fmt in $FMT_PREFIX"
fi

# Decide lib dir (Home-built fmt on macOS will usually use lib, not lib64)
if [ -d "$FMT_PREFIX/lib" ]; then
  FMT_LIB="$FMT_PREFIX/lib"
elif [ -d "$FMT_PREFIX/lib64" ]; then
  FMT_LIB="$FMT_PREFIX/lib64"
else
  echo "ERROR: Could not find libfmt library under $FMT_PREFIX/lib or lib64" >&2
  exit 1
fi

FMT_INCLUDE="$FMT_PREFIX/include"

echo ">>> fmt include: $FMT_INCLUDE"
echo ">>> fmt lib:     $FMT_LIB"

# ===== Step 2: Clone mkdssp (dssp) source =====
cd "$SRC_ROOT"

if [ ! -d dssp ]; then
  echo ">>> Cloning PDB-REDO/dssp"
  git clone https://github.com/PDB-REDO/dssp.git
fi

cd dssp
echo ">>> Checking out tag $DSSP_TAG"
git fetch --tags
git checkout "$DSSP_TAG"

# ===== Step 3: Configure mkdssp build with fmt =====
echo ">>> Cleaning CMake and CPM caches"
rm -rf build
rm -rf "${CPM_SOURCE_CACHE:-$HOME/.cache/CPM}"

echo ">>> Configuring mkdssp with CMake"
"$CMAKE" -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_INSTALL_PREFIX="$DSSP_PREFIX" \
  "-DCMAKE_CXX_FLAGS=-I${FMT_INCLUDE}" \
  "-DCMAKE_CXX_STANDARD_LIBRARIES=-L${FMT_LIB} -lfmt"

# ===== Step 4: Build & install mkdssp =====
echo ">>> Building mkdssp"
"$CMAKE" --build build -j"$(sysctl -n hw.ncpu)"

echo ">>> Installing mkdssp into $DSSP_PREFIX"
"$CMAKE" --install build

# ===== Step 5: Smoke test =====
echo ">>> mkdssp installed. To use it, add this to your shell profile:"
echo
echo "    export PATH=\"$DSSP_PREFIX/bin:\$PATH\""
echo

if [ -x "$DSSP_PREFIX/bin/mkdssp" ]; then
  echo ">>> mkdssp version:"
  "$DSSP_PREFIX/bin/mkdssp" --version || true
else
  echo "WARNING: mkdssp not found at $DSSP_PREFIX/bin/mkdssp" >&2
fi

