# -------- Base Python (multi-arch friendly) --------
ARG PYTHON_VERSION=3.10.14
FROM python:${PYTHON_VERSION}-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- build deps for DSSP ----
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cmake make g++ \
      libboost-dev zlib1g-dev libbz2-dev libeigen3-dev libpcre2-dev \
      libcifpp-dev libmcfp-dev \
      ca-certificates curl git tini \
      build-essential pkg-config \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      vim-tiny \
      && ln -sf /usr/bin/vim.tiny /usr/bin/vi \
      && vi --version | head -n1 \
      && rm -rf /var/lib/apt/lists/*

# ---- build & install mkdssp v4.5.3 ----
ARG DSSP_VER=v4.5.3
RUN git clone --depth=1 --branch ${DSSP_VER} https://github.com/PDB-REDO/dssp.git /tmp/dssp \
 && cmake -S /tmp/dssp -B /tmp/dssp/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /tmp/dssp/build -j"$(nproc)" \
 && cmake --install /tmp/dssp/build \
 && rm -rf /tmp/dssp

# sanity check (leave or remove)
RUN mkdssp --version && command -v mkdssp

# -------- Python deps (CPU-only) --------
COPY requirements.txt /tmp/requirements.txt

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
    mkdir -p /app /workspace /logs && \
    chown -R app:app /workspace /home/app /logs && \
    chmod 0555 /app && \
    chmod 0775 /workspace && \
    chmod 0775 /logs

VOLUME ["/workspace"]
WORKDIR /workspace
USER app

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES=""

ENTRYPOINT ["/usr/bin/tini","-g","--"]
CMD ["python","--version"]

