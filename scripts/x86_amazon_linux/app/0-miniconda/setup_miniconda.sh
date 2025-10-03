#!/bin/bash
set -e

# 1) Install Miniconda (skip if already installed)
if [ ! -d "$HOME/miniconda3" ]; then
  echo ">>> Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda3
  rm miniconda.sh
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
  echo ">>> Miniconda already installed."
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
$HOME/miniconda3/bin/conda init bash

# 2) Create and activate a conda environment with Python 3.11
if ! conda env list | grep -q "base311"; then
  echo ">>> Creating conda env base311 with Python 3.11..."
  conda create -y -n base311 python=3.11
fi
conda --version
conda activate base311

echo ">>> Done. To use:"
echo "    deactivate                                  # Exit venv"

