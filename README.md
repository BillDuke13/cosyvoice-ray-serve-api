# CosyVoice Ray Serve API

Ray Serve HTTP service for [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`). Exposes FastAPI endpoints for built-in voice synthesis, zero-shot cloning, cross-lingual synthesis, and instruction-controlled synthesis with buffered or chunked MP3 output.

The first-party service lives in `cosyvoice_api/`. The vendored `cosyvoice/` package is pinned to upstream commit `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` and kept read-only.

## Requirements

- **Python 3.11** (pinned in `.python-version`, provisioned automatically by `uv`)
- **[uv](https://docs.astral.sh/uv/)** for environment and lockfile management
- **FFmpeg**, **Sox**, and **libsndfile** on `PATH`
- **NVIDIA GPU (optional, recommended)**: Linux installs PyTorch and TorchCodec wheels from the CUDA 13.2 (`cu132`) index (Turing through Blackwell `sm_120`) and `onnxruntime-gpu` for CUDA 13. Requires NVIDIA driver `>= 580` (`>= 595.45.04` for CUDA 13.2 GA). CPU fallback works automatically when CUDA is unavailable.

## Quick Start

### Local

1. Install system audio packages:

   ```bash
   # Ubuntu / Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg sox libsox-fmt-all libsndfile1

   # macOS
   brew install ffmpeg sox libsndfile
   ```

2. Install Python dependencies from `uv.lock`:

   ```bash
   uv sync
   ```

3. Run the service against an existing Ray cluster or start a local single-node cluster:

   ```bash
   # Existing Ray cluster
   uv run serve run cosyvoice_api:app

   # Local single-node Ray cluster (binds to SERVE_PORT, default 8000)
   uv run python -m cosyvoice_api
   ```

   On first startup, `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` downloads from ModelScope into `pretrained_models/Fun-CosyVoice3-0.5B/` (~10 GB).

4. Synthesize speech:

   ```bash
   curl -X POST http://localhost:8000/tts \
     -H 'Content-Type: application/json' \
     -d '{"text": "Hello, world.", "voice_type": "qwen"}' \
     --output hello.mp3
   ```

### Docker

```bash
docker build -t cosyvoice-ray-serve-api .

# GPU (requires NVIDIA Container Toolkit)
docker run -d --gpus all -p 9998:9998 -p 8265:8265 --name cosyvoice cosyvoice-ray-serve-api

# CPU
docker run -d -p 9998:9998 --name cosyvoice cosyvoice-ray-serve-api
```

The container sets `SERVE_PORT=9998` and checks `GET /v1/model/cosyvoice/healthcheck`.

## Endpoints

All synthesis routes return `audio/mpeg`, support optional `"stream": true` for chunked streaming, and accept `speed` in `[0.5, 2.0]` (default `1.0`).

| Method | Path | Body | Description |
| --- | --- | --- | --- |
| `POST` | `/tts` | JSON | Synthesize text using a built-in voice prompt (`voice_type`) |
| `POST` | `/zero_shot_tts` | `multipart/form-data` | Clone a voice from an uploaded clip and transcript |
| `POST` | `/cross_lingual_tts` | `multipart/form-data` | Clone a voice from an uploaded clip without a transcript |
| `POST` | `/instruct_tts` | JSON | Style/emotion synthesis via a natural-language `instruction` |
| `GET` | `/health` | - | Liveness/readiness check (aliased at `/v1/model/cosyvoice/healthcheck`) |
| `GET` | `/config` | - | Active voice, sample rate, device, and model metadata |

See [docs/api.md](docs/api.md) for request schemas, error codes, and `curl` examples.

## Repository Layout

```text
cosyvoice-ray-serve-api/
├── cosyvoice_api/          # First-party Ray Serve + FastAPI service
│   ├── __init__.py         # Public exports (`app`, `CosyVoiceService`, `Settings`)
│   ├── __main__.py         # Local runner (`python -m cosyvoice_api`)
│   ├── audio.py            # Tensor validation/normalization, FFmpeg MP3 encoding, voice cache
│   ├── config.py           # Environment-driven Settings and DeploymentSettings
│   ├── deployment.py       # @serve.deployment and FastAPI @serve.ingress binding
│   ├── engine.py           # GPU selection, ModelScope download, AutoModel inference wrapper
│   ├── prompts.py          # CosyVoice3 `<|endofprompt|>` formatting and speed validation
│   └── service.py          # HTTP route handlers and request lifecycle
├── assets/voices/          # Built-in voice prompt WAVs (default: qwen.wav)
├── scripts/entrypoint.sh   # Container entrypoint (Ray head + `python -m cosyvoice_api`)
├── docs/                   # API, architecture, and deployment documentation
├── tests/                  # Unit and FastAPI handler tests (mocked model/FFmpeg)
├── cosyvoice/              # Vendored upstream CosyVoice source (read-only)
├── Dockerfile              # Multi-stage uv + CUDA 13.2 container build
├── pyproject.toml          # Dependencies, uv index config, ruff and pytest settings
├── uv.lock                 # Locked dependency graph
└── .python-version         # Pins Python 3.11 for uv
```

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

> `transformers` is pinned to `4.51.3` in `pyproject.toml` to avoid CosyVoice3 Qwen2 attention-mask regressions in `transformers >= 4.53 / 5.x` (upstream issues `#1886`, `#1953`) while keeping `cosyvoice/` unmodified.

## Documentation

- [API Reference](docs/api.md) — endpoint contracts, request fields, status codes, and examples
- [Architecture](docs/architecture.md) — module boundaries, inference mapping, and audio pipeline
- [Deployment Guide](docs/deployment.md) — local setup, Docker/CUDA 13.2, environment variables, and scaling

## License

Apache 2.0. See [LICENSE](LICENSE). Upstream CosyVoice code and model weights are subject to their respective licenses.
