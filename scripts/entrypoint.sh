#!/usr/bin/env bash
# Container entrypoint: starts a single-node Ray head and runs cosyvoice_api.
set -euo pipefail

RAY_PID=""
API_PID=""

log() {
    local level="$1"
    shift
    printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$*"
}

detect_memory_bytes() {
    local host_bytes
    host_bytes="$(free -b | awk '/^Mem:/{print $2}')"

    # Prefer cgroup v2 memory cap when set below host RAM.
    if [ -r /sys/fs/cgroup/memory.max ]; then
        local cgroup_max
        cgroup_max="$(cat /sys/fs/cgroup/memory.max)"
        if [ "$cgroup_max" != "max" ] && [[ "$cgroup_max" =~ ^[0-9]+$ ]] && [ "$cgroup_max" -lt "$host_bytes" ]; then
            echo "$cgroup_max"
            return 0
        fi
    fi

    echo "$host_bytes"
}

check_resources() {
    log INFO "Inspecting container resources"

    NUM_CPUS="$(nproc)"
    log INFO "CPUs: ${NUM_CPUS}"

    TOTAL_MEMORY_BYTES="$(detect_memory_bytes)"
    MEMORY_GB="$((TOTAL_MEMORY_BYTES / 1024 / 1024 / 1024))"
    MEMORY_AVAILABLE_GB="$(free -g | awk '/^Mem:/{print $7}')"
    log INFO "Memory: ${MEMORY_AVAILABLE_GB}GB host available / ${MEMORY_GB}GB effective limit"

    NUM_GPUS=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_names
        if gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
            NUM_GPUS="$(printf '%s\n' "$gpu_names" | sed '/^$/d' | wc -l | tr -d ' ')"
            if [ "$NUM_GPUS" -gt 0 ]; then
                log INFO "GPUs detected: ${NUM_GPUS}"
                nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while read -r line; do
                    log INFO "  GPU: ${line}"
                done
            else
                log WARN "No GPUs detected; running on CPU"
            fi
        else
            log WARN "nvidia-smi returned an error; running on CPU"
        fi
    else
        log WARN "nvidia-smi not available; running on CPU"
    fi

    DISK_USAGE="$(df -h /app | awk 'NR==2 {print $5}')"
    log INFO "Disk usage (/app): ${DISK_USAGE}"
}

start_ray() {
    log INFO "Starting single-node Ray head"

    # Allocate 80% of effective memory to Ray and 25% of that to the object store.
    MEMORY_BYTES="$((TOTAL_MEMORY_BYTES * 80 / 100))"
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
    RAY_PID=$!

    local timeout=60
    local elapsed=0

    log INFO "Waiting for Ray cluster readiness"
    while ! ray status >/dev/null 2>&1; do
        if ! kill -0 "$RAY_PID" 2>/dev/null; then
            log ERROR "Ray head process exited unexpectedly"
            exit 1
        fi
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

    # Keep bash as PID 1 so SIGTERM/SIGINT runs cleanup() cleanly on docker stop.
    python -m cosyvoice_api &
    API_PID=$!
    wait "$API_PID"
}

cleanup() {
    log INFO "Received shutdown signal; stopping API and Ray"
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill -TERM "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    ray stop --force >/dev/null 2>&1 || true
    if [ -n "$RAY_PID" ] && kill -0 "$RAY_PID" 2>/dev/null; then
        wait "$RAY_PID" 2>/dev/null || true
    fi
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
