# API Reference

HTTP endpoints exposed by the FastAPI ingress in `cosyvoice_api/deployment.py` (handlers implemented in `cosyvoice_api/service.py`).

- Base URL: `http://<host>:<SERVE_PORT>` (`SERVE_PORT` defaults to `8000` locally and `9998` in Docker).
- Synthesis responses return `Content-Type: audio/mpeg` (24 kHz model audio encoded to LAME VBR MP3).
- Pass `"stream": true` (JSON) or `stream=true` (form field) on any synthesis endpoint for chunked streaming.
- Optional `speed` accepts any finite float in `[0.5, 2.0]` (default `1.0`).

## Synthesis Endpoints

### `POST /tts`

Synthesizes text with a built-in voice by routing the bundled reference clip through `inference_cross_lingual`.

Body (`application/json`):

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `text` | string | yes | - | Text to synthesize |
| `voice_type` | string | no | `qwen` | Key from `VOICE_PROMPTS`; unknown values fall back to the default voice |
| `speed` | number | no | `1.0` | Finite float in `[0.5, 2.0]` |
| `stream` | boolean | no | `false` | Stream MP3 chunks as they are generated |

```bash
curl -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, world.", "voice_type": "qwen", "speed": 1.0}' \
  --output out.mp3
```

### `POST /zero_shot_tts`

Clones a speaker from an uploaded reference clip and its transcript (`inference_zero_shot`). If `reference_text` does not contain `<|endofprompt|>`, it is wrapped with the CosyVoice3 system prompt automatically.

Body (`multipart/form-data`):

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `text` | string | yes | Text to synthesize |
| `reference_text` | string | yes | Transcript of `reference_audio` |
| `reference_audio` | file | yes | Audio clip (WAV/MP3/FLAC/...); normalized to 16 kHz mono WAV |
| `speed` | number | no | Finite float in `[0.5, 2.0]` (default `1.0`) |
| `stream` | boolean | no | `"true"` to stream chunks |

```bash
curl -X POST http://localhost:8000/zero_shot_tts \
  -F 'text=This voice is cloned.' \
  -F 'reference_text=Transcript of the sample.' \
  -F 'reference_audio=@sample.wav' \
  --output out.mp3
```

### `POST /cross_lingual_tts`

Clones a speaker from an uploaded reference clip without requiring a transcript (`inference_cross_lingual`). Target text may be in a different language than the reference audio.

Body (`multipart/form-data`):

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `text` | string | yes | Target text in any supported language |
| `reference_audio` | file | yes | Audio clip; normalized to 16 kHz mono WAV |
| `speed` | number | no | Finite float in `[0.5, 2.0]` (default `1.0`) |
| `stream` | boolean | no | `"true"` to stream chunks |

```bash
curl -X POST http://localhost:8000/cross_lingual_tts \
  -F 'text=你好，世界。' \
  -F 'reference_audio=@sample.wav' \
  --output out.mp3
```

### `POST /instruct_tts`

Steers style, dialect, or emotion using a natural-language instruction over a built-in voice (`inference_instruct2`). Plain `instruction` values without `<|endofprompt|>` are wrapped as `You are a helpful assistant. {instruction}<|endofprompt|>`.

Body (`application/json`):

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `text` | string | yes | - | Text to synthesize |
| `instruction` | string | yes | - | Style, dialect, or emotion instruction |
| `voice_type` | string | no | `qwen` | Built-in voice providing timbre |
| `speed` | number | no | `1.0` | Finite float in `[0.5, 2.0]` |
| `stream` | boolean | no | `false` | Stream MP3 chunks |

```bash
curl -X POST http://localhost:8000/instruct_tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Today is a wonderful day.", "instruction": "Speak cheerfully."}' \
  --output out.mp3
```

## Operational Endpoints

### `GET /health`

Aliased at `GET /v1/model/cosyvoice/healthcheck` (used by the Docker `HEALTHCHECK`). Returns `200` when the model is loaded and `503` otherwise.

```json
{
  "status": "healthy",
  "timestamp": "2026-11-26T12:00:00+0000",
  "device": "cuda:0",
  "model_loaded": true,
  "ray_actor_id": "...",
  "ray_node_id": "..."
}
```

### `GET /config`

Returns the active replica configuration:

```json
{
  "default_voice": "qwen",
  "available_voice_prompts": ["qwen"],
  "sample_rate": 24000,
  "reference_audio_sample_rate": 16000,
  "device_in_use": "cuda:0",
  "model_directory": "/app/pretrained_models/Fun-CosyVoice3-0.5B",
  "model_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
  "model_dir_name": "Fun-CosyVoice3-0.5B",
  "model_revision": "master",
  "model_family": "cosyvoice3",
  "max_replicas_config": 2
}
```

## Error Responses

Errors return JSON `{"detail": "<message>"}`:

| Status | Cause |
| --- | --- |
| `400` | Invalid JSON/multipart form, missing/blank fields, invalid `speed`/`stream`, unavailable voice prompt, or invalid/corrupt reference audio |
| `413` | Uploaded `reference_audio` exceeds `MAX_REFERENCE_AUDIO_BYTES` (default 25 MiB) |
| `500` | Internal synthesis or audio-encoding failure |
| `503` | Model not loaded (`/health` or synthesis requests before readiness) |

Uploaded reference audio clips are capped at 30 seconds during 16 kHz mono WAV normalization to match CosyVoice3's frontend limit.

CORS defaults to `allow_origins=["*"]` with `allow_credentials=False`. Set `CORS_ALLOW_ORIGINS` (comma-separated origins) to restrict browser access in production.
