#!/usr/bin/env bash
# Container entrypoint: starts a single-node Ray head and runs cosyvoice_api.
set -euo pipefail

log() {
    local level="$1"
    shift
    printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$*"
}

check_resources() {
    log INFO "Inspecting container resources"

    NUM_CPUS="$(nproc)"
    log INFO "CPUs: ${NUM_CPUS}"

    MEMORY_GB="$(free -g | awk '/^Mem:/{print $2}')"
    MEMORY_AVAILABLE_GB="$(free -g | awk '/^Mem:/{print $7}')"
    log INFO "Memory: ${MEMORY_AVAILABLE_GB}GB available / ${MEMORY_GB}GB total"

    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)"
        if [ "$NUM_GPUS" -gt 0 ]; then
            log INFO "GPUs detected: ${NUM_GPUS}"
            nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read -r line; do
                log INFO "  GPU: ${line}"
            done
        else
            log WARN "No GPUs detected; running on CPU"
        fi
    else
        log WARN "nvidia-smi not available; running on CPU"
        NUM_GPUS=0
    fi

    DISK_USAGE="$(df -h /app | awk 'NR==2 {print $5}')"
    log INFO "Disk usage (/app): ${DISK_USAGE}"
}

start_ray() {
    log INFO "Starting single-node Ray head"

    # Allocate 80% of container memory to Ray and 25% of that to the object store.
    MEMORY_BYTES="$(($(free -b | awk '/^Mem:/{print $2}') * 80 / 100))"
    OBJECT_STORE_MEMORY="$((MEMORY_BYTES / 4))"

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
    local elapsed=0

    log INFO "Waiting for Ray cluster readiness"
    while ! ray status >/dev/null 2>&1; do
        if [ "$elapsed" -ge "$timeout" ]; then
            log ERROR "Ray failed to start within ${timeout}s"
            exit 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    log INFO "Ray cluster is ready"
    ray status
}

start_api() {
    log INFO "Starting CosyVoice Ray Serve API"

    export RAY_MEMORY="$((MEMORY_BYTES * 90 / 100))"
    export RAY_OBJECT_STORE_MEMORY="$OBJECT_STORE_MEMORY"

    exec python -m cosyvoice_api
}

cleanup() {
    log INFO "Received shutdown signal; stopping Ray"
    ray stop --force || true
    exit 0
}

main() {
    if [ "$EUID" -eq 0 ]; then
        log INFO "Dropping root privileges to user cosyvoice"
        exec gosu cosyvoice "$0" "$@"
    fi

    check_resources
    start_ray
    start_api
}

trap cleanup SIGTERM SIGINT

main "$@"
