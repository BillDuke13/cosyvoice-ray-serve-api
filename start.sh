#!/bin/bash
# Container entrypoint: start a single-node Ray cluster, then run the Ray Serve app.
# Extracted from the Dockerfile so the startup logic is readable and maintainable.
set -euo pipefail

# Color codes for output
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

echo -e "${BLUE}🚀 Starting CosyVoice Ray Serve API${NC}"

# Log with timestamp
log() {
    echo -e "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

# Report CPU / memory / GPU / disk; sets NUM_CPUS and NUM_GPUS for start_ray.
check_resources() {
    log "${BLUE}📊 Checking system resources...${NC}"

    NUM_CPUS=$(nproc)
    log "CPUs: $NUM_CPUS"

    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    MEMORY_AVAILABLE_GB=$(free -g | awk '/^Mem:/{print $7}')
    log "Memory: ${MEMORY_AVAILABLE_GB}GB available / ${MEMORY_GB}GB total"

    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
        if [ "$NUM_GPUS" -gt 0 ]; then
            log "${GREEN}GPUs detected: $NUM_GPUS${NC}"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read -r line; do
                log "  GPU: $line"
            done
        else
            log "${YELLOW}⚠️  No GPUs detected${NC}"
        fi
    else
        log "${YELLOW}⚠️  nvidia-smi not available${NC}"
        NUM_GPUS=0
    fi

    DISK_USAGE=$(df -h /app | awk 'NR==2 {print $5}')
    log "Disk usage: $DISK_USAGE"

    return 0
}

# Start the Ray head node; sets MEMORY_BYTES and OBJECT_STORE_MEMORY for start_api.
start_ray() {
    log "${BLUE}🔧 Starting Ray cluster...${NC}"

    # Allocate 80% of total memory to Ray; 25% of that to the object store.
    MEMORY_BYTES=$(($(free -b | awk '/^Mem:/{print $2}') * 80 / 100))
    OBJECT_STORE_MEMORY=$((MEMORY_BYTES / 4))

    ray start --head \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --num-cpus="$NUM_CPUS" \
        --num-gpus="$NUM_GPUS" \
        --memory="$MEMORY_BYTES" \
        --object-store-memory="$OBJECT_STORE_MEMORY" \
        --plasma-directory=/tmp \
        --temp-dir=/app/tmp \
        --block &

    local timeout=60
    local counter=0

    log "Waiting for Ray to be ready..."
    while ! ray status >/dev/null 2>&1; do
        if [ $counter -ge $timeout ]; then
            log "${RED}❌ Ray failed to start within ${timeout} seconds${NC}"
            exit 1
        fi
        printf "."
        sleep 1
        counter=$((counter + 1))
    done

    echo
    log "${GREEN}✅ Ray cluster is ready!${NC}"
    ray status
}

# Launch the Ray Serve application.
start_api() {
    log "${BLUE}🎤 Starting CosyVoice API server...${NC}"

    export RAY_MEMORY=$((MEMORY_BYTES * 90 / 100))
    export RAY_OBJECT_STORE_MEMORY=$OBJECT_STORE_MEMORY

    exec python api.py
}

main() {
    # Step down from root to the unprivileged service user.
    if [ "$EUID" -eq 0 ]; then
        log "Switching to non-root user..."
        exec gosu cosyvoice "$0" "$@"
    fi

    check_resources
    start_ray
    start_api
}

cleanup() {
    log "${YELLOW}🛑 Shutting down gracefully...${NC}"
    ray stop --force || true
    exit 0
}

trap cleanup SIGTERM SIGINT

main "$@"
