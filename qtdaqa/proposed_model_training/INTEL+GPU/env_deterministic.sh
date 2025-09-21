#!/usr/bin/env bash
# Source this script prior to launching run_training.sh to enforce deterministic
# behaviour on CUDA.

export PYTHONHASHSEED=222
export PL_SEED_WORKERS=1
export TORCH_USE_DETERMINISTIC_ALGORITHMS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_TF32_OVERRIDE=0
export TORCH_BACKEND_CUDA_FUSER_DISABLE=1
export OMP_NUM_THREADS=8
