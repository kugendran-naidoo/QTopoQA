# -------- Base Python (multi-arch friendly) --------
ARG PYTHON_VERSION=3.10.14
FROM python:${PYTHON_VERSION}-slim-bookworm

ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG TORCHAUDIO_VERSION=2.1.2
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
# PyG index keyed to torch-2.1.x CPU
ARG PYG_EXTRA_INDEX=https://data.pyg.org/whl/torch-2.1.0+cpu.html

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# -------- OS deps --------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git tini \
        build-essential pkg-config \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        dssp \
        vim-tiny \
        && ln -sf /usr/bin/vim.tiny /usr/bin/vi \
        && vi --version | head -n1 \
        && rm -rf /var/lib/apt/lists/* \
    # verify mkdssp is available (fail the build if not)
        && mkdssp --version \
        && command -v mkdssp

# -------- Python deps (CPU-only) --------
COPY requirements.txt /tmp/requirements.txt

# ---- Python deps (Torch/PyG/PL + project) ----
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
ARG PYG_EXTRA_INDEX=https://data.pyg.org/whl/torch-2.1.0+cpu.html
ARG TORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG TORCHAUDIO_VERSION=2.1.2

RUN <<'SH'
set -eu

if [ -s /tmp/requirements.txt ]; then
  # Normalize CRLF → LF
  tr -d '\r' < /tmp/requirements.txt > /tmp/requirements.lf.txt

  # Exclude Torch/PyG lines already handled earlier
  awk 'BEGIN{IGNORECASE=1} \
       !/^(torch|torchvision|torchaudio|pyg-lib|torch-scatter|torch-sparse|torch-cluster|torch-spline-conv|torch-geometric)\b/ {print}' \
       /tmp/requirements.lf.txt > /tmp/requirements.project.txt

  # Install project deps (if any)
  if [ -s /tmp/requirements.project.txt ]; then
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.project.txt
  fi
fi
SH
# === [END ADDED] ===



# -------- User & permissions --------
RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g app -s /bin/bash app && \
    mkdir -p /app /workspace && \
    chown -R app:app /workspace /home/app && \
    chmod 0555 /app && \
    chmod 0775 /workspace

VOLUME ["/workspace"]
WORKDIR /workspace
USER app

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES=""

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["python","--version"]

