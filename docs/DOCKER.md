# CosyVoice Docker Implementation

This document provides technical details about the Docker implementation for the CosyVoice Ray Serve API.

## Overview

The Docker implementation enables containerized deployment of the CosyVoice Ray Serve API, providing a consistent and isolated environment for running the service. The implementation is based on NVIDIA's CUDA container for GPU support.

## Dockerfile Structure

The Dockerfile is structured in a multi-stage build process:

1. **Base Image**: Uses NVIDIA CUDA 11.8 with cuDNN 8 on Ubuntu 22.04
2. **Environment Setup**: Configures environment variables for optimal performance
3. **System Dependencies**: Installs required system packages
4. **User Setup**: Creates a non-root user for security
5. **Python Dependencies**: Installs Python packages
6. **Application Code**: Copies the application code
7. **Runtime Configuration**: Sets up ports, health checks, and startup commands

## Key Components

### Base Image

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder
```

This base image provides:
- CUDA 11.8 for GPU acceleration
- cuDNN 8 for deep learning operations
- Ubuntu 22.04 as the base operating system

### Environment Variables

```dockerfile
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
```

These environment variables configure:
- Non-interactive package installation
- Python behavior for containerized environments
- Ray cluster settings
- CUDA configuration for optimal GPU usage
- PyTorch memory allocation settings

### System Dependencies

```dockerfile
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
```

Installs required system packages:
- Python 3.11 with development headers
- FFmpeg for audio processing
- Sox for sound processing
- Build tools for compiling dependencies
- Curl for health checks

### User Setup

```dockerfile
RUN useradd -m -u 1000 ray && \
    mkdir -p /app /data/ray && \
    chown -R ray:ray /app /data/ray
```

Creates a non-root user for security:
- Creates a 'ray' user with UID 1000
- Creates application and data directories
- Sets appropriate permissions

### Python Dependencies

```dockerfile
RUN python3.11 -m pip install --no-cache-dir -U pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt && \
    python3.11 -m pip install --no-cache-dir ray[serve] torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

Installs Python packages:
- Updates pip to the latest version
- Installs dependencies from requirements.txt
- Installs Ray Serve for API serving
- Installs PyTorch and TorchAudio with CUDA support

### Application Code

```dockerfile
COPY --chown=ray:ray . .
```

Copies the application code:
- Transfers all files from the build context
- Sets appropriate ownership to the 'ray' user

### Runtime Configuration

```dockerfile
EXPOSE 9998 8265 6379

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9998/-/healthz || exit 1

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
```

Configures the container runtime:
- Exposes ports for the API (9998), Ray dashboard (8265), and Ray cluster (6379)
- Sets up a health check that verifies the API is responding
- Defines the startup command that:
  1. Starts a Ray cluster
  2. Waits for the cluster to initialize
  3. Starts the CosyVoice service using Ray Serve

## Docker Compose (Optional)

A Docker Compose configuration can simplify deployment:

```yaml
version: '3.8'

services:
  cosyvoice-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: cosyvoice-ray-serve-api:latest
    container_name: cosyvoice-api
    restart: unless-stopped
    ports:
      - "9998:9998"
      - "8265:8265"
    volumes:
      - ./asset:/app/asset
      - ./logs:/app/logs
      - ./tmp:/app/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9998/-/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

## Building and Running

### Building the Docker Image

```bash
docker build -t cosyvoice-ray-serve-api .
```

### Running the Container

```bash
docker run -d --name cosyvoice-api \
  --gpus all \
  -p 9998:9998 \
  -p 8265:8265 \
  -v $(pwd)/asset:/app/asset \
  -v $(pwd)/logs:/app/logs \
  cosyvoice-ray-serve-api
```

This command:
- Runs the container in detached mode
- Enables all available GPUs
- Maps the API and dashboard ports
- Mounts volumes for assets and logs

## Volume Mounts

The container uses several volume mounts:

1. **asset**: Contains voice prompt files
   - Path: `/app/asset`
   - Purpose: Provides voice samples for synthesis

2. **logs**: Contains application logs
   - Path: `/app/logs`
   - Purpose: Persists logs for monitoring and debugging

3. **tmp** (optional): Temporary file storage
   - Path: `/app/tmp`
   - Purpose: Stores temporary files during processing

## GPU Support

The container is configured for NVIDIA GPU support:

1. Uses the NVIDIA Container Toolkit
2. Requires NVIDIA drivers on the host
3. Passes all available GPUs to the container
4. Configures CUDA for optimal performance

## Health Monitoring

The container includes a health check that:

1. Runs every 30 seconds
2. Verifies the API is responding
3. Allows 3 retries before marking unhealthy
4. Provides a 5-second startup grace period

## Security Considerations

The Docker implementation includes several security measures:

1. **Non-root User**: Runs as a non-privileged 'ray' user
2. **Minimal Dependencies**: Installs only required packages
3. **Clean Image**: Removes package lists after installation
4. **Isolated Environment**: Provides process isolation

## Performance Optimization

The container is optimized for performance:

1. **CUDA Configuration**: Optimized settings for GPU usage
2. **Memory Allocation**: Configured for efficient memory usage
3. **CPU Utilization**: Automatically detects and uses available CPUs
4. **Caching Disabled**: Prevents unnecessary disk usage

## Troubleshooting

Common issues and solutions:

1. **GPU Not Detected**
   - Ensure NVIDIA drivers are installed on the host
   - Verify the NVIDIA Container Toolkit is installed
   - Check that `--gpus all` is included in the run command

2. **Container Exits Immediately**
   - Check logs with `docker logs cosyvoice-api`
   - Ensure Ray can start properly
   - Verify port availability

3. **API Not Responding**
   - Check if the container is running
   - Verify port mappings
   - Check logs for initialization errors

4. **Out of Memory Errors**
   - Adjust `PYTORCH_CUDA_ALLOC_CONF` environment variable
   - Limit concurrent requests
   - Consider using a host with more GPU memory
