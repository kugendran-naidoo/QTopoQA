#!/bin/bash
set -e

echo "=== Installing mkdssp via Homebrew ==="

# Add the bio tap if not already added
brew tap brewsci/bio || true

# Install DSSP (provides mkdssp binary)
brew install dssp

echo
echo "=== Verifying mkdssp installation ==="

# Check if mkdssp is on PATH and print version
if command -v mkdssp >/dev/null 2>&1; then
    echo "mkdssp installed at: $(which mkdssp)"
    mkdssp --version
else
    echo "mkdssp not found in PATH"
    exit 1
fi

echo
echo "mkdssp installation complete"

