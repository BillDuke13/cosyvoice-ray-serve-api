# syntax=docker/dockerfile:1.7
# Multi-stage build for the CosyVoice3 Ray Serve API.
#
# Dependencies are managed by uv (pyproject.toml + uv.lock). The CUDA 13.0 base
# supports Blackwell / RTX 50-series (sm_120) via the torch cu130 wheels pinned in
# pyproject's [tool.uv.index]. There is no conda/pynini layer: CosyVoice3 uses the
# wetext text-normalization path, which does not require pynini.

ARG CUDA_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# ---------- builder: install the locked dependency tree into /app/.venv ----------
FROM ${CUDA_IMAGE} AS builder

# Pinned uv binary (copied from the official distroless image).
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.10 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Toolchain for any sdist lacking a cp310 wheel; discarded with this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ONLY dependencies (cached unless uv.lock / pyproject.toml change).
# pyproject sets package=false, so the project itself is not installed; api.py and
# the vendored cosyvoice/ package are imported as local files at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --locked --no-dev --no-install-project

# ---------- runtime ----------
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# NVIDIA Container settings
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0"

# CUDA / PyTorch performance settings (arch list includes 12.0 = Blackwell sm_120).
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0" \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    CUBLAS_WORKSPACE_CONFIG=:16:8

# Ray configuration for Docker
ENV RAY_DISABLE_DOCKER_CPU_WARNING=1 \
    RAY_DEDUP_LOGS=0 \
    RAY_memory_monitor_refresh_ms=5000 \
    RAY_memory_usage_threshold=0.90 \
    RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 \
    RAY_enable_gpu_detection=1

# Application configuration (names must match the env vars read in api.py)
ENV SERVE_PORT=9998 \
    COSYVOICE_MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    COSYVOICE_MODEL_DIR_NAME=Fun-CosyVoice3-0.5B \
    COSYVOICE_MODEL_REVISION=master \
    MAX_ONGOING_REQUESTS_PER_REPLICA=2 \
    MAX_QUEUED_REQUESTS_DEPLOYMENT=10 \
    MAX_REPLICAS=4

# Make the uv-managed virtual environment the default Python.
ENV PATH="/app/.venv/bin:$PATH"

# System runtime libraries: audio stack (ffmpeg/sox/libsndfile), health utils, gosu.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    curl \
    ca-certificates \
    procps \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Non-root service user
RUN groupadd -r cosyvoice && useradd -r -g cosyvoice -u 1000 cosyvoice

WORKDIR /app

# uv-managed interpreter + the synced virtual environment from the builder stage.
# The venv references the interpreter under /python, so both must be copied.
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv

# Application code: api.py, vendored cosyvoice/, asset/*.wav prompts, start.sh.
COPY --chown=cosyvoice:cosyvoice . /app

# Writable runtime directories (model cache, temp audio, logs).
RUN mkdir -p /app/tmp /app/logs /app/pretrained_models /tmp/cuda_cache \
    && chown -R cosyvoice:cosyvoice /app/tmp /app/logs /app/pretrained_models /tmp/cuda_cache \
    && chmod +x /app/start.sh

# Expose serve API, Ray dashboard, Ray GCS, Ray client.
EXPOSE 9998 8265 6379 10001

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVE_PORT}/v1/model/cosyvoice/healthcheck || exit 1

USER cosyvoice

ENTRYPOINT ["/app/start.sh"]
