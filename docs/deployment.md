# Deployment Guide

Runtime behavior is controlled through environment variables read by `cosyvoice_api/config.py`; there is no separate configuration file.

## Prerequisites

- **Python 3.11** (pinned via `.python-version`, provisioned automatically by `uv`).
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**:
  ```bash
  # Linux / macOS standalone installer
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or macOS Homebrew
  brew install uv
  ```
- **System audio libraries**: `ffmpeg`, `sox`, and `libsndfile`.
- **GPU (optional)**: Linux pulls `torch`, `torchaudio`, and `torchcodec` from the PyTorch CUDA 13.2 (`cu132`) index (Turing through Blackwell `sm_120`) and installs `onnxruntime-gpu` for CUDA 13. Host requires NVIDIA driver `>= 580` (`>= 595.45.04` for CUDA 13.2 GA).
- **Disk space**: ~10 GB under `pretrained_models/Fun-CosyVoice3-0.5B/` for weights downloaded from ModelScope on first startup.

## Local Execution

```bash
# Ubuntu / Debian system dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg sox libsox-fmt-all libsndfile1

# Sync locked Python 3.11 environment
uv sync
```

Launch with either command:

| Command | Behavior |
| --- | --- |
| `uv run serve run cosyvoice_api:app` | Deploys onto an existing Ray cluster |
| `uv run python -m cosyvoice_api` | Initializes a local single-node Ray cluster and serves on `SERVE_PORT` (default `8000`) |

## Docker

`Dockerfile` builds a multi-stage image on `nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04` using `ghcr.io/astral-sh/uv:0.13.10`. The builder stage runs `uv sync --locked --no-dev --no-install-project`; the runtime stage copies `/python` and `/app/.venv`, runs as non-root user `cosyvoice`, and starts via `scripts/entrypoint.sh` (single-node Ray head + `python -m cosyvoice_api`).

```bash
docker build -t cosyvoice-ray-serve-api .

# GPU (requires NVIDIA Container Toolkit)
docker run -d --gpus all \
  -p 9998:9998 \
  -p 8265:8265 \
  -v /path/to/pretrained_models:/app/pretrained_models \
  --name cosyvoice \
  cosyvoice-ray-serve-api

# CPU
docker run -d -p 9998:9998 --name cosyvoice cosyvoice-ray-serve-api
```

Exposed ports:
- `9998`: HTTP API (`SERVE_PORT` default in the container)
- `8265`: Ray Dashboard
- `6379`: Ray GCS
- `10001`: Ray Client

Built-in voice prompts in `assets/voices/*.wav` are copied into the image (preserved by `.dockerignore`). Model weights are excluded and either mounted into `/app/pretrained_models` or downloaded on first start.

## Environment Variables

| Variable | Local Default | Docker Default | Purpose |
| --- | --- | --- | --- |
| `COSYVOICE_MODEL_ID` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | same | ModelScope repository ID |
| `COSYVOICE_MODEL_DIR_NAME` | `Fun-CosyVoice3-0.5B` | same | Subdirectory under `pretrained_models/` |
| `COSYVOICE_MODEL_REVISION` | `master` | same | ModelScope revision passed to `snapshot_download` |
| `SERVE_PORT` | `8000` | `9998` | HTTP proxy bind port (`python -m cosyvoice_api`) |
| `MIN_REPLICAS` | `1` | `1` | Autoscaling minimum replicas |
| `INITIAL_REPLICAS` | `1` | `1` | Initial replicas at deployment startup |
| `MAX_REPLICAS` | `2` | `4` | Autoscaling maximum replicas |
| `TARGET_ONGOING_REQUESTS` | `2` | `2` | Target concurrent requests per replica |
| `MAX_ONGOING_REQUESTS_PER_REPLICA` | `5` | `2` | Hard concurrency limit per replica |
| `MAX_QUEUED_REQUESTS_DEPLOYMENT` | `20` | `10` | Maximum queued requests across deployment |
| `NUM_GPUS_PER_REPLICA` | `1` | `1` | GPUs requested per replica (`0` forces CPU) |
| `HEALTH_CHECK_PERIOD_S` | `15` | `15` | Ray Serve health check interval (seconds) |
| `HEALTH_CHECK_TIMEOUT_S` | `30` | `30` | Ray Serve health check timeout (seconds) |
| `GRACEFUL_SHUTDOWN_TIMEOUT_S` | `60` | `60` | Drain timeout on replica shutdown (seconds) |
| `GRACEFUL_SHUTDOWN_WAIT_LOOP_S` | `5` | `5` | Drain poll interval (seconds) |
| `MAX_REFERENCE_AUDIO_BYTES` | `26214400` (25 MiB) | same | Max uploaded reference audio size (bytes; returns `413` if exceeded) |
| `CORS_ALLOW_ORIGINS` | `*` | `*` | Comma-separated allowed CORS origins |
| `CUDA_VISIBLE_DEVICES` | unset | unset | Restrict visible NVIDIA GPUs (`""` forces CPU) |

## Adding Built-In Voices

Place a `.wav` file under `assets/voices/` and register its `voice_type` key in `VOICE_PROMPTS` inside `cosyvoice_api/config.py`.

## Health, Logging, and Scaling

- **Probes**: `GET /health` and `GET /v1/model/cosyvoice/healthcheck` return `200` once loaded, `503` otherwise.
- **Logs**: Written to `stdout` (`docker logs <container>`) and `logs/cosyvoice_api.log`.
- **Scaling**: Each GPU replica loads its own copy of CosyVoice3, so `MAX_REPLICAS` should not exceed available GPUs when `NUM_GPUS_PER_REPLICA=1`. Enable `"stream": true` on longer prompts to reduce time-to-first-byte.

## Troubleshooting

- **Model download fails**: confirm outbound network access to ModelScope and write permissions on `pretrained_models/`.
- **FFmpeg or libsndfile errors**: confirm `ffmpeg`, `sox`, and `libsndfile` are installed and reachable on `PATH`.
- **CUDA initialization errors**: check host NVIDIA driver (`>= 580`), NVIDIA Container Toolkit setup, and `CUDA_VISIBLE_DEVICES`. To force CPU execution explicitly, set `NUM_GPUS_PER_REPLICA=0` and `CUDA_VISIBLE_DEVICES=""`.
