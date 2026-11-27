# syntax=docker/dockerfile:1.7
# Multi-stage image for the CosyVoice3 Ray Serve API.
# Uses CUDA 13.2 (Blackwell / RTX 50-series sm_120) and uv-managed Python 3.11.

ARG CUDA_IMAGE=nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

# ---------- builder: install locked dependencies into /app/.venv ----------
FROM ${CUDA_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.11 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Build toolchain for sdists without prebuilt wheels (e.g. matcha-tts); omitted from runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sync dependencies only so this layer stays cached across application edits.
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

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=13.0"

# Target architectures from Turing (7.5) through Blackwell (12.0).
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=0 \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0" \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    CUBLAS_WORKSPACE_CONFIG=:16:8

ENV RAY_DISABLE_DOCKER_CPU_WARNING=1 \
    RAY_DEDUP_LOGS=0 \
    RAY_memory_monitor_refresh_ms=5000 \
    RAY_memory_usage_threshold=0.90 \
    RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 \
    RAY_enable_gpu_detection=1

# Application defaults (names match cosyvoice_api/config.py).
ENV SERVE_PORT=9998 \
    COSYVOICE_MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    COSYVOICE_MODEL_DIR_NAME=Fun-CosyVoice3-0.5B \
    COSYVOICE_MODEL_REVISION=master \
    MAX_ONGOING_REQUESTS_PER_REPLICA=2 \
    MAX_QUEUED_REQUESTS_DEPLOYMENT=10 \
    MAX_REPLICAS=4

ENV PATH="/app/.venv/bin:$PATH"

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

RUN groupadd -r cosyvoice && useradd -r -g cosyvoice -u 1000 cosyvoice

WORKDIR /app

# Both /python and /app/.venv are required because the venv symlinks the managed interpreter.
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv

COPY --chown=cosyvoice:cosyvoice . /app

RUN mkdir -p /app/tmp /app/logs /app/pretrained_models /tmp/cuda_cache \
    && chown -R cosyvoice:cosyvoice /app/tmp /app/logs /app/pretrained_models /tmp/cuda_cache \
    && chmod +x /app/scripts/entrypoint.sh

EXPOSE 9998 8265 6379 10001

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVE_PORT}/v1/model/cosyvoice/healthcheck || exit 1

USER cosyvoice

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
