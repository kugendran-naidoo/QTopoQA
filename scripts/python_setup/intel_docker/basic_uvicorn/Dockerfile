# syntax=docker/dockerfile:1.7-labs
# Build a minimal AMD64 image pinned to Python 3.10.8.
# Works when built from macOS using buildx (cross-compiling to linux/amd64).

ARG PYTHON_VERSION=3.10.8
FROM --platform=linux/amd64 python:${PYTHON_VERSION}-slim-bullseye

# System settings & safety
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install useful system deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl bash tini \
    && rm -rf /var/lib/apt/lists/*

# Create app user and app directory (make sure /app exists before chown)
RUN useradd -m -u 1000 appuser \
 && mkdir -p /app \
 && chown -R appuser:appuser /app

# Make /app writable by appuser BEFORE switching user or creating the venv
RUN chown -R appuser:appuser /app

WORKDIR /app
USER appuser

# Switch to non-root
USER appuser

# copy app and install deps:
COPY --chown=appuser:appuser requirements.txt .

# Create venv in /app and install deps with a pip cache mount
# NOTE: --mount must come immediately after RUN
RUN --mount=type=cache,target=/home/appuser/.cache/pip \
    python -m venv .venv && \
    . .venv/bin/activate && \
    ./.venv/bin/pip install --upgrade pip && \
    ./.venv/bin/pip install -r requirements.txt

# Copy the rest of your app (owned by appuser)
COPY --chown=appuser:appuser . .

ENV PATH="/app/.venv/bin:${PATH}"

# Expose a default command that shows the Python version; override as needed.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "app.py"]

