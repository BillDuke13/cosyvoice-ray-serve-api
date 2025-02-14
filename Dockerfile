# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RAY_TEMP_DIR=/data/ray \
    RAY_ADDRESS="ray://localhost:6379" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="all" \
    CUDA_DEVICE_ORDER="PCI_BUS_ID" \
    CUDA_LAUNCH_BLOCKING="1" \
    PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    sox \
    libsox-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and user
RUN useradd -m -u 1000 ray && \
    mkdir -p /app /data/ray && \
    chown -R ray:ray /app /data/ray

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY --chown=ray:ray requirements.txt .

# Install Python dependencies
RUN python3.11 -m pip install --no-cache-dir -U pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt && \
    python3.11 -m pip install --no-cache-dir ray[serve] torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY --chown=ray:ray . .

# Switch to non-root user
USER ray

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/asset /app/pretrained_models

# Runtime configuration
EXPOSE 9998 8265 6379

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9998/-/healthz || exit 1

# Start Ray and CosyVoice service
CMD ray start --head \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --port=6379 \
        --temp-dir=/data/ray \
        --num-cpus=$(nproc) \
        --block & \
    sleep 5 && \
    echo "Starting CosyVoice service..." && \
    ray serve run \
        --address=$RAY_ADDRESS \
        --namespace=cosyvoice \
        cosyvoice.api:deployment
