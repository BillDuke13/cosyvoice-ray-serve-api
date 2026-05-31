---
name: smoke-test
description: Smoke-test a running CosyVoice Ray Serve API — check /health and /config, send a /tts request, and save the returned MP3. Use to manually verify the service works after a change, since the repo has no automated tests.
---

# Smoke-test the CosyVoice API

Exercise a running instance of this service end to end. There is no automated test suite, so this is the primary way to confirm a change actually works.

## Inputs

`$ARGUMENTS` may specify a base URL and/or test text. Defaults:
- Base URL: `http://localhost:8000`
- Text: a short mixed English + Chinese sentence.

## Steps

1. **Confirm the service is up.** If it isn't already running, ask whether to start it (`serve run api:cosyvoice_app`, or `python api.py` for a local cluster) or whether the user will point at a running instance. Don't start a deploy unprompted.
2. **Health and config.**
   - `curl -fsS <base>/health`
   - `curl -fsS <base>/config`
   Report the status and the active configuration.
3. **Standard TTS.** POST to `<base>/tts` with a JSON body. Read the `tts_standard` handler in `api.py` for the exact request shape (field names, required voice/params) — do not guess. Save the response to `tmp/smoke_tts.mp3`.
4. **Verify output.** Confirm the file is non-empty and a valid MP3 (e.g. `ffprobe tmp/smoke_tts.mp3`, or check the magic bytes). Report duration if available.
5. **Report.** Summarize which endpoints responded, their HTTP statuses, the output file path and size, and any error bodies. On failure, surface the response body and the relevant tail of `logs/cosyvoice_api.log`.

## Notes

- Output lands in `tmp/` (gitignored).
- The request schema for every endpoint lives in the corresponding `CosyVoiceService` method in `api.py` — read it rather than inventing field names.
