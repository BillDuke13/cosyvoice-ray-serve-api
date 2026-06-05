# Multi-stage Docker build for CosyVoice Ray Serve API
# Based on NVIDIA CUDA with Ubuntu 22.04 for optimal GPU performance

# Build stage
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    python3.10-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python build dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Production stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables for production
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_LAUNCH_BLOCKING=0 \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# NVIDIA Container settings
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.1"

# CUDA optimization settings
ENV CUDA_CACHE_PATH=/tmp/cuda_cache \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    CUBLAS_WORKSPACE_CONFIG=:16:8

# Ray configuration for Docker
ENV RAY_DISABLE_DOCKER_CPU_WARNING=1 \
    RAY_SCHEDULER_EVENTS=0 \
    RAY_DEDUP_LOGS=0 \
    RAY_memory_monitor_refresh_ms=5000 \
    RAY_memory_usage_threshold=0.90 \
    RAY_serve_http_memory_ratio=0.15 \
    RAY_serve_http_memory=1000000000 \
    RAY_serve_http_max_concurrent_queries=200 \
    RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 \
    RAY_enable_gpu_detection=1

# Application configuration (names must match the env vars read in api.py)
ENV SERVE_PORT=9998 \
    COSYVOICE_MODEL_ID=FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    COSYVOICE_MODEL_DIR_NAME=Fun-CosyVoice3-0.5B \
    COSYVOICE_MODEL_REVISION=main \
    MAX_ONGOING_REQUESTS_PER_REPLICA=2 \
    MAX_QUEUED_REQUESTS_DEPLOYMENT=10 \
    MAX_REPLICAS=4

# Create non-root user for security
RUN groupadd -r cosyvoice && useradd -r -g cosyvoice -u 1000 cosyvoice

# Set working directory
WORKDIR /app

# Install system dependencies and audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    libsox-fmt-all \
    libsox-dev \
    # System utilities
    curl \
    wget \
    ca-certificates \
    # Python runtime
    python3.10 \
    python3.10-distutils \
    python3-pip \
    # Health check utilities
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install Miniconda for better package management
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create symlinks for Python
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python && \
    ln -sf /opt/conda/bin/python /usr/local/bin/python3 && \
    ln -sf /opt/conda/bin/pip /usr/local/bin/pip

# Install pynini via conda for text normalization dependencies that are not reliably
# available as portable PyPI wheels across deployment targets.
RUN conda install -y -c conda-forge pynini==2.1.5 && \
    conda clean -afy

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt && \
    pip cache purge

# Create application directories with proper permissions
RUN mkdir -p /app/tmp /app/logs /app/asset /app/pretrained_models /tmp/cuda_cache && \
    chown -R cosyvoice:cosyvoice /app && \
    chmod 755 /app/tmp /app/logs /app/asset /app/pretrained_models /tmp/cuda_cache

# Copy application code
COPY --chown=cosyvoice:cosyvoice . .

# Create startup script with health monitoring
RUN printf '#!/bin/bash\n\
set -euo pipefail\n\
\n\
# Color codes for output\n\
RED="\\033[0;31m"\n\
GREEN="\\033[0;32m"\n\
YELLOW="\\033[1;33m"\n\
BLUE="\\033[0;34m"\n\
NC="\\033[0m" # No Color\n\
\n\
echo -e "${BLUE}🚀 Starting CosyVoice Ray Serve API${NC}"\n\
\n\
# Function to log with timestamp\n\
log() {\n\
    echo -e "[$(date "+%%Y-%%m-%%d %%H:%%M:%%S")] $1"\n\
}\n\
\n\
# Function to check system resources\n\
check_resources() {\n\
    log "${BLUE}📊 Checking system resources...${NC}"\n\
    \n\
    # CPU info\n\
    NUM_CPUS=$(nproc)\n\
    log "CPUs: $NUM_CPUS"\n\
    \n\
    # Memory info\n\
    MEMORY_GB=$(free -g | awk "/^Mem:/{print \\$2}")\n\
    MEMORY_AVAILABLE_GB=$(free -g | awk "/^Mem:/{print \\$7}")\n\
    log "Memory: ${MEMORY_AVAILABLE_GB}GB available / ${MEMORY_GB}GB total"\n\
    \n\
    # GPU info\n\
    if command -v nvidia-smi >/dev/null 2>&1; then\n\
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")\n\
        if [ "$NUM_GPUS" -gt 0 ]; then\n\
            log "${GREEN}GPUs detected: $NUM_GPUS${NC}"\n\
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do\n\
                log "  GPU: $line"\n\
            done\n\
        else\n\
            log "${YELLOW}⚠️  No GPUs detected${NC}"\n\
        fi\n\
    else\n\
        log "${YELLOW}⚠️  nvidia-smi not available${NC}"\n\
        NUM_GPUS=0\n\
    fi\n\
    \n\
    # Disk space\n\
    DISK_USAGE=$(df -h /app | awk "NR==2 {print \\$5}")\n\
    log "Disk usage: $DISK_USAGE"\n\
    \n\
    return 0\n\
}\n\
\n\
# Function to start Ray cluster\n\
start_ray() {\n\
    log "${BLUE}🔧 Starting Ray cluster...${NC}"\n\
    \n\
    # Calculate memory allocation (80%% of available)\n\
    MEMORY_BYTES=$(($(free -b | awk "/^Mem:/{print \\$2}") * 80 / 100))\n\
    OBJECT_STORE_MEMORY=$((MEMORY_BYTES / 4))  # 25%% for object store\n\
    \n\
    # Start Ray head node\n\
    ray start --head \\\n\
        --port=6379 \\\n\
        --dashboard-host=0.0.0.0 \\\n\
        --dashboard-port=8265 \\\n\
        --num-cpus="$NUM_CPUS" \\\n\
        --num-gpus="$NUM_GPUS" \\\n\
        --memory="$MEMORY_BYTES" \\\n\
        --object-store-memory="$OBJECT_STORE_MEMORY" \\\n\
        --plasma-directory=/tmp \\\n\
        --temp-dir=/app/tmp \\\n\
        --block &\n\
    \n\
    # Wait for Ray to be ready\n\
    local timeout=60\n\
    local counter=0\n\
    \n\
    log "Waiting for Ray to be ready..."\n\
    while ! ray status >/dev/null 2>&1; do\n\
        if [ $counter -ge $timeout ]; then\n\
            log "${RED}❌ Ray failed to start within ${timeout} seconds${NC}"\n\
            exit 1\n\
        fi\n\
        printf "."\n\
        sleep 1\n\
        counter=$((counter + 1))\n\
    done\n\
    \n\
    echo\n\
    log "${GREEN}✅ Ray cluster is ready!${NC}"\n\
    ray status\n\
}\n\
\n\
# Function to start CosyVoice API\n\
start_api() {\n\
    log "${BLUE}🎤 Starting CosyVoice API server...${NC}"\n\
    \n\
    # Set Ray memory limits based on available memory\n\
    export RAY_MEMORY=$(($MEMORY_BYTES * 90 / 100))\n\
    export RAY_OBJECT_STORE_MEMORY=$OBJECT_STORE_MEMORY\n\
    \n\
    # Start the API\n\
    exec python api.py\n\
}\n\
\n\
# Main execution\n\
main() {\n\
    # Change to non-root user\n\
    if [ "$EUID" -eq 0 ]; then\n\
        log "Switching to non-root user..."\n\
        exec gosu cosyvoice "$0" "$@"\n\
    fi\n\
    \n\
    # Check resources\n\
    check_resources\n\
    \n\
    # Start Ray\n\
    start_ray\n\
    \n\
    # Start API\n\
    start_api\n\
}\n\
\n\
# Handle shutdown signals gracefully\n\
cleanup() {\n\
    log "${YELLOW}🛑 Shutting down gracefully...${NC}"\n\
    ray stop --force || true\n\
    exit 0\n\
}\n\
\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Run main function\n\
main "$@"\n\
' > /app/start.sh && \
    chmod +x /app/start.sh

# Install gosu for step-down from root
RUN wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/1.17/gosu-$(dpkg --print-architecture)" && \
    chmod +x /usr/local/bin/gosu && \
    gosu nobody true

# Expose ports (serve API, Ray dashboard, Ray GCS, Ray client)
EXPOSE 9998 8265 6379 10001

# Health check with more comprehensive testing
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${SERVE_PORT}/v1/model/cosyvoice/healthcheck || exit 1

# Security: Use non-root user
USER cosyvoice

# Entry point
ENTRYPOINT ["/app/start.sh"]
