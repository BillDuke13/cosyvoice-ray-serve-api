# CosyVoice Ray Serve API

A [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) HTTP API that serves the
[CosyVoice3](https://github.com/FunAudioLLM/CosyVoice) text-to-speech model
(`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`). It wraps the model in a scalable service with
autoscaling worker replicas, automatic GPU detection, reference-audio caching, MP3 output,
optional response streaming, and health endpoints for orchestration.

The entire service -- the `CosyVoiceService` deployment, the `AudioProcessor` helpers, the
Starlette routing, and the `cosyvoice_app` entry point -- lives in a single module, `api.py`.
The `cosyvoice/` package is vendored from upstream CosyVoice commit
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` and is treated as read-only.

> Detailed, browsable documentation lives in [`docs/`](docs/index.html) (HTML): API reference,
> architecture, and deployment guide.

## Features

- **Standard TTS with a built-in voice** (`POST /tts`). The service clones a bundled
  reference clip selected by `voice_type` via the model's prompt-based cross-lingual path.
- **Zero-shot voice cloning** (`POST /zero_shot_tts`) from an uploaded reference clip plus its
  transcript.
- **Cross-lingual synthesis** (`POST /cross_lingual_tts`) from an uploaded reference clip,
  with no transcript required; the target text may be in another language.
- **Instruction-controlled synthesis** (`POST /instruct_tts`) with a natural-language style,
  dialect, speed, or emotion instruction over a built-in voice.
- **CosyVoice3 prompt compatibility**. Plain legacy client text is wrapped internally with
  CosyVoice3's `<|endofprompt|>` marker where the model requires it.
- **Streaming**. Pass `"stream": true` for a chunked `audio/mpeg` response on any synthesis
  endpoint.
- **Adjustable speed**. Optional `speed` factor (finite float, `0.5` to `2.0`;
  `1.0` = normal) on every synthesis endpoint.
- **MP3 output**. 24 kHz model audio is encoded to MP3 via FFmpeg (LAME VBR).
- **Scalability and reliability**. Ray Serve autoscaling, per-replica GPU assignment with CPU
  fallback, graceful shutdown, and health checks.

## Requirements

- **Python 3.10**
- **[uv](https://docs.astral.sh/uv/)** for dependency management (`pyproject.toml` + `uv.lock`).
- **FFmpeg**, **Sox**, and **libsndfile** for audio I/O and format conversion.
- **GPU (optional, recommended)**. PyTorch wheels come from the **CUDA 13.0** (`cu130`) index
  (configured in `pyproject.toml`), supporting Turing through Blackwell / RTX 50-series
  (sm_120). The service also runs on CPU. Minimum NVIDIA driver: **580.65.06** for CUDA 13.

## Installation

### Option 1: Local

1. **Clone:**

   ```bash
   git clone https://github.com/BillDuke13/cosyvoice-ray-serve-api.git
   cd cosyvoice-ray-serve-api
   ```

2. **Install system dependencies:**

   Ubuntu/Debian:

   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg sox libsox-fmt-all libsndfile1
   ```

   macOS (Homebrew):

   ```bash
   brew install ffmpeg sox libsndfile
   ```

3. **Install Python dependencies:**

   ```bash
   uv sync
   ```

   This creates `.venv/` and installs the exact dependency tree from `uv.lock`. Python 3.10
   is pinned via `.python-version` and managed by uv automatically.

4. **Model download.** On first startup, `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` downloads
   from ModelScope into `pretrained_models/Fun-CosyVoice3-0.5B/` (gitignored; requires
   network and about 10 GB of model files).

### Option 2: Docker

```bash
docker build -t cosyvoice-ray-serve-api .
```

The image uses a multi-stage uv build on `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`,
installs the audio and Python dependencies, and runs as a non-root user.

## Running

### Locally

Against an existing Ray cluster:

```bash
uv run serve run api:cosyvoice_app
```

Or start a local single-node Ray cluster directly:

```bash
uv run python api.py
```

Both bind the HTTP proxy to `SERVE_PORT` (`8000` by default for local runs). The API is then
available at `http://localhost:8000`.

### With Docker

The image sets `SERVE_PORT=9998`, so publish that port (and optionally the Ray dashboard on
8265):

```bash
# CPU
docker run -d -p 9998:9998 --name cosyvoice cosyvoice-ray-serve-api

# GPU (requires the NVIDIA Container Toolkit)
docker run -d --gpus all -p 9998:9998 -p 8265:8265 --name cosyvoice cosyvoice-ray-serve-api
```

The API is then available at `http://localhost:9998`. The container `HEALTHCHECK` curls
`http://localhost:9998/v1/model/cosyvoice/healthcheck`.

## API endpoints

All synthesis endpoints return an MP3 (`audio/mpeg`). Add `"stream": true` for a chunked
streaming response, and an optional `speed` (finite float, `0.5` to `2.0`; `1.0` = normal).

### `POST /tts` -- standard TTS

JSON body:

```json
{ "text": "Hello, world.", "voice_type": "qwen", "speed": 1.0, "stream": false }
```

`voice_type` selects a built-in voice from `VOICE_PROMPTS` (defaults to `qwen`); only `text` is
required.

### `POST /zero_shot_tts` -- zero-shot voice cloning

`multipart/form-data`:

| Field | Required | Description |
| --- | --- | --- |
| `text` | yes | Text to synthesize |
| `reference_text` | yes | Transcript of the reference clip |
| `reference_audio` | yes | Reference audio file (WAV/MP3/...) |
| `speed` | no | Speed factor (`0.5` to `2.0`) |
| `stream` | no | `true` for streaming |

If `reference_text` does not include `<|endofprompt|>`, the service wraps it in the CosyVoice3
prompt format before inference.

### `POST /cross_lingual_tts` -- cross-lingual cloning

`multipart/form-data`:

| Field | Required | Description |
| --- | --- | --- |
| `text` | yes | Text to synthesize (may be another language than the reference) |
| `reference_audio` | yes | Reference audio file |
| `speed` | no | Speed factor (`0.5` to `2.0`) |
| `stream` | no | `true` for streaming |

### `POST /instruct_tts` -- instruction-controlled TTS

JSON body:

```json
{
  "text": "Today is a wonderful day.",
  "instruction": "Speak cheerfully and energetically.",
  "voice_type": "qwen",
  "speed": 1.0,
  "stream": false
}
```

`text` and `instruction` are required; `voice_type` selects the built-in voice that supplies
the timbre. If `instruction` does not include `<|endofprompt|>`, the service wraps it as
`You are a helpful assistant. {instruction}<|endofprompt|>`.

### `GET /health`

Alias: `GET /v1/model/cosyvoice/healthcheck`.

Returns `200` with `{"status": "healthy", "device": ..., "model_loaded": true, ...}` when the
model is loaded, or `503` `{"status": "unhealthy", ...}` otherwise.

### `GET /config`

Returns the active configuration: default voice, available voice prompts, sample rates, the
device in use, the model directory, and CosyVoice3 model metadata.

## Configuration

Behavior is driven entirely by environment variables read in the `@serve.deployment` decorator
and `reconfigure()` in `api.py`; there is no config file.

| Variable | Default | Purpose |
| --- | --- | --- |
| `COSYVOICE_MODEL_ID` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ModelScope model ID |
| `COSYVOICE_MODEL_DIR_NAME` | `Fun-CosyVoice3-0.5B` | Local directory under `pretrained_models/` |
| `COSYVOICE_MODEL_REVISION` | `master` | Model revision passed to `snapshot_download` |
| `SERVE_PORT` | `8000` (Docker: `9998`) | HTTP proxy bind port (used by `python api.py`) |
| `MIN_REPLICAS` | `1` | Autoscaling lower bound |
| `INITIAL_REPLICAS` | `1` | Replicas at startup |
| `MAX_REPLICAS` | `2` (Docker: `4`) | Autoscaling upper bound |
| `TARGET_ONGOING_REQUESTS` | `2` | Target concurrent requests per replica |
| `MAX_ONGOING_REQUESTS_PER_REPLICA` | `5` | Max concurrent requests per replica |
| `MAX_QUEUED_REQUESTS_DEPLOYMENT` | `20` | Max queued requests at the deployment |
| `NUM_GPUS_PER_REPLICA` | `1` | GPUs per replica (`0` forces CPU) |
| `HEALTH_CHECK_PERIOD_S` | `15` | Health-check interval |
| `HEALTH_CHECK_TIMEOUT_S` | `30` | Health-check timeout |
| `GRACEFUL_SHUTDOWN_TIMEOUT_S` | `60` | Time to finish in-flight requests on shutdown |
| `GRACEFUL_SHUTDOWN_WAIT_LOOP_S` | `5` | Shutdown completion poll interval |
| `CUDA_VISIBLE_DEVICES` | - | Standard NVIDIA GPU-visibility control |

### Built-in voices

`VOICE_PROMPTS` in `api.py` maps a `voice_type` key to a `.wav` in `asset/` (default:
`qwen -> asset/qwen.wav`). These clips are used as the cloning prompt for `/tts` and
`/instruct_tts`. To add a voice, drop a `.wav` into `asset/` and register it in
`VOICE_PROMPTS`. Audio sample rates are fixed and load-bearing: model output is 24 kHz,
reference/prompt audio is normalized to 16 kHz mono.

## Project structure

```text
cosyvoice-ray-serve-api/
├── api.py             # Ray Serve app: service, audio processing, routing, entry point
├── Dockerfile         # Multi-stage uv build on CUDA 13.0 (nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04)
├── start.sh           # Container entrypoint: starts Ray cluster then launches the Serve app
├── pyproject.toml     # Project metadata and uv dependency configuration
├── uv.lock            # Locked dependency tree (source of truth for all Python deps)
├── .python-version    # Pins Python 3.10 for uv
├── ruff.toml          # Format/lint config (target py310; excludes cosyvoice/)
├── asset/             # Built-in voice prompt clips (e.g. qwen.wav)
├── cosyvoice/         # Vendored upstream CosyVoice source (read-only)
├── docs/              # HTML documentation
├── pretrained_models/ # Auto-downloaded model weights (gitignored)
├── tmp/               # Temporary audio (created at runtime, gitignored)
└── logs/              # Log files (created at runtime, gitignored)
```

## Logging

Logs go to `stdout` (visible via `docker logs <container>`) and to
`logs/cosyvoice_api.log`.

## Testing

Format and lint with Ruff:

```bash
ruff format --check .
ruff check .
python -m py_compile api.py
```

For service validation, start the service and exercise `/health`, `/config`, and each synthesis
endpoint with `stream=false` and `stream=true`. Model smoke tests require the CosyVoice3 model
files and can be slow on first startup.

## Troubleshooting

- **Model download fails** -- ensure network access on first run and write permission for
  `pretrained_models/`.
- **FFmpeg/Sox/libsndfile not found** -- confirm all three are installed and on `PATH` or present
  in the container.
- **CUDA errors** -- verify host NVIDIA drivers are >= 580.65.06 (required for CUDA 13); with
  Docker, confirm the NVIDIA Container Toolkit is installed; check `CUDA_VISIBLE_DEVICES`. The
  service falls back to CPU if GPU initialization fails.
- **`uv sync` fails** -- ensure uv is installed (`pip install uv` or see
  [docs.astral.sh/uv](https://docs.astral.sh/uv/)) and that the system packages above are
  present before syncing.

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details. The
underlying CosyVoice project and model have their own licensing terms, which should also be
respected.

## Acknowledgements

- The [FunAudioLLM team](https://github.com/FunAudioLLM) for creating and open-sourcing
  CosyVoice.
- The [Ray Serve team](https://docs.ray.io/en/latest/serve/index.html) for the serving
  framework.
