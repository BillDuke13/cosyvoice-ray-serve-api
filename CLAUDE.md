# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Ray Serve application that serves the CosyVoice2 text-to-speech model over HTTP. The whole service — the `CosyVoiceService` deployment, the `AudioProcessor`, the route wiring, and the app object — lives in a single module, `api.py` (~1470 lines). `cosyvoice/` is the vendored CosyVoice2 model source. Human-facing HTML docs (API reference, architecture, deployment) live in `docs/`.

## Running the service

- `serve run api:cosyvoice_app` — run against an existing Ray cluster (the documented entry point).
- `python api.py` — start a local single-node Ray cluster and serve on `0.0.0.0:8000` (the `__main__` block; for quick local debugging).
- Docker build/run commands are in `README.md`.

The Ray Serve entry point is the module attribute `cosyvoice_app` in `api.py` (a Starlette app). The HTTP bind port comes from `SERVE_PORT` (defaults to **8000** for local `python api.py`; the Docker image sets `SERVE_PORT=9998`).

## HTTP endpoints

`POST /tts`, `POST /zero_shot_tts`, `POST /cross_lingual_tts`, `POST /instruct_tts`, `GET /health` (aliased as `GET /v1/model/cosyvoice/healthcheck`, the Docker `HEALTHCHECK` target), `GET /config`. Routes are wired in the Starlette `app` near the bottom of `api.py`; each maps to a `CosyVoiceService` method. Responses are MP3; pass `"stream": true` in the request body for chunked streaming, and an optional `speed` float. Read the corresponding service method for each endpoint's exact request shape rather than guessing field names.

CosyVoice2-0.5B ships no fine-tuned (SFT) speakers, so the endpoints map to its inference methods as follows: `/tts` → `inference_cross_lingual` with a built-in prompt clip from `asset/` (selected by `voice_type`); `/zero_shot_tts` → `inference_zero_shot` (uploaded clip + transcript); `/cross_lingual_tts` → `inference_cross_lingual` (uploaded clip); `/instruct_tts` → `inference_instruct2` with a built-in prompt clip. Don't route `/tts` through `inference_sft` (no speaker bank) or `/instruct_tts` through `inference_instruct` (raises `NotImplementedError` on CosyVoice2).

## Vendored code

Treat `cosyvoice/` as read-only upstream code (the CosyVoice2 source, imported as `cosyvoice.cli.cosyvoice.CosyVoice2`). Do not modify it. Make changes in `api.py` and other project-root files instead.

## Environment & setup

- Python 3.10.
- `pynini==2.1.5` must be installed from conda-forge **before** `pip install -r requirements.txt` (it is a C-extension not available on PyPI):
  `conda install -y -c conda-forge pynini==2.1.5 && pip install -r requirements.txt`
- System packages required: `ffmpeg` and `sox` / `libsox-dev` — audio I/O and format conversion run as `ffmpeg` subprocess calls.
- GPU wheels are pinned to CUDA 12.1 (`torch==2.3.1` via the cu121 index, `onnxruntime-gpu`, `tensorrt-cu12`). Keep these aligned when changing any one of them.
- On first startup the model `iic/CosyVoice2-0.5B` auto-downloads via ModelScope into `pretrained_models/CosyVoice2-0.5B/` (gitignored; needs network, can be slow).

## Configuration

Deployment behavior is driven entirely by environment variables read in the `@serve.deployment` decorator and `reconfigure()` in `api.py` (replica autoscaling, `NUM_GPUS_PER_REPLICA`, health-check and graceful-shutdown timeouts, default voice). There is no config file — change defaults in `api.py`.

## Code style

- Format and lint with Ruff: `ruff format .` and `ruff check --fix .` (config in `ruff.toml`, target `py310`). `cosyvoice/` is excluded.
- Audio sample rates are fixed and load-bearing: model output is 24 kHz (`SAMPLE_RATE`), reference audio is normalized to 16 kHz mono (`REFERENCE_AUDIO_SAMPLE_RATE`). Don't change these casually.

## Ports & health checks

HTTP binds to `SERVE_PORT` (default 8000 locally, 9998 in Docker). The Docker `HEALTHCHECK` curls `http://localhost:${SERVE_PORT}/v1/model/cosyvoice/healthcheck`, which `api.py` serves as an alias of `/health`. Keep these in sync if you change the port or health route.

## Testing

There is no automated test suite or CI. Verify changes by running the service and exercising the endpoints — see the `/smoke-test` skill. If you add tests, no runner is configured yet; propose the setup first.
