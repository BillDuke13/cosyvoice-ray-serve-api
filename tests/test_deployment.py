from __future__ import annotations

import asyncio
import gc
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from cosyvoice_api import app as ray_app
from cosyvoice_api import http_app
from cosyvoice_api.audio import AudioProcessingError
from cosyvoice_api.config import Settings
from cosyvoice_api.deployment import CosyVoiceService
from cosyvoice_api.http_utils import TempPathGuard
from cosyvoice_api.service import CosyVoiceHandlers, create_asgi_app


class FakeEngine:
    def __init__(
        self,
        *,
        loaded: bool = True,
        device: str = "cpu",
        fail_mid_stream: bool = False,
    ) -> None:
        self._loaded = loaded
        self.device = device
        self.fail_mid_stream = fail_mid_stream
        self.calls: list[tuple[str, str, bool, dict[str, Any]]] = []
        self.closed = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def iter_speech_chunks(
        self,
        method_name: str,
        text: str,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> Iterator[torch.Tensor]:
        self.calls.append((method_name, text, stream, kwargs))
        prompt_wav = kwargs.get("prompt_wav")
        assert prompt_wav is not None and Path(prompt_wav).exists()
        yield torch.tensor([[0.1, -0.2, 0.3]])
        if self.fail_mid_stream:
            raise RuntimeError("synthetic mid-stream failure")
        yield torch.tensor([[0.4, -0.5, 0.6]])

    def synthesize(self, method_name: str, text: str, **kwargs: Any) -> torch.Tensor:
        chunks = list(self.iter_speech_chunks(method_name, text, stream=False, **kwargs))
        return torch.cat(chunks, dim=-1)

    def close(self) -> None:
        self.closed = True
        self._loaded = False


async def _fake_mp3_encoder(tensor: torch.Tensor, _tmp_dir: Path, _sr: int) -> bytes:
    return f"MP3:{tensor.shape[-1]}:".encode()


async def _fake_ref_normalizer(src: Path, dst: Path, _sr: int) -> Path:
    assert src.exists()
    dst.write_bytes(b"WAV16K")
    return dst


def _build_test_client(
    tmp_path: Path,
    *,
    loaded: bool = True,
    env: dict[str, str] | None = None,
    ref_normalizer: Any = _fake_ref_normalizer,
    fail_mid_stream: bool = False,
) -> tuple[TestClient, FakeEngine, CosyVoiceHandlers, Settings]:
    settings = Settings.from_env(env or {}, root_dir=tmp_path)
    settings.ensure_runtime_dirs()
    settings.voice_prompts["qwen"].write_bytes(b"RIFF_QWEN")
    engine = FakeEngine(loaded=loaded, fail_mid_stream=fail_mid_stream)
    handlers = CosyVoiceHandlers(
        settings=settings,
        engine=engine,
        mp3_encoder=_fake_mp3_encoder,
        ref_normalizer=ref_normalizer,
    )
    client = TestClient(create_asgi_app(handlers))
    return client, engine, handlers, settings


def test_ray_serve_ingress_frozen_asgi_app_end_to_end(tmp_path: Path) -> None:
    """Exercise the real cloudpickled @serve.ingress(http_app) class across all routes."""
    assert ray_app is not None
    assert http_app is not None

    settings = Settings.from_env({}, root_dir=tmp_path)
    settings.ensure_runtime_dirs()
    settings.voice_prompts["qwen"].write_bytes(b"RIFF_QWEN")
    engine = FakeEngine(loaded=True)

    cls = CosyVoiceService.func_or_class
    instance = cls.__new__(cls)

    async def _init() -> None:
        await instance.__init__(
            settings=settings,
            engine=engine,
            mp3_encoder=_fake_mp3_encoder,
            ref_normalizer=_fake_ref_normalizer,
        )

    asyncio.run(_init())
    # Must generate OpenAPI schema without ForwardRef errors on the cloudpickled app.
    schema_paths = instance._asgi_app.openapi()["paths"]
    for expected_path in (
        "/health",
        "/v1/model/cosyvoice/healthcheck",
        "/config",
        "/tts",
        "/zero_shot_tts",
        "/cross_lingual_tts",
        "/instruct_tts",
    ):
        assert expected_path in schema_paths

    with patch(
        "ray.serve.get_replica_context",
        return_value=SimpleNamespace(servable_object=instance),
    ):
        client = TestClient(instance._asgi_app)
        assert client.get("/health").status_code == 200
        assert client.get("/v1/model/cosyvoice/healthcheck").status_code == 200
        assert client.get("/config").status_code == 200
        assert client.post("/tts", json={"text": "Hello"}).content == b"MP3:6:"
        assert (
            client.post(
                "/instruct_tts",
                json={"text": "Hi", "instruction": "Speak warmly."},
            ).content
            == b"MP3:6:"
        )
        assert (
            client.post(
                "/zero_shot_tts",
                data={"text": "Clone", "reference_text": "Ref"},
                files={"reference_audio": ("sample.wav", b"RIFF_DATA", "audio/wav")},
            ).content
            == b"MP3:6:"
        )
        assert (
            client.post(
                "/cross_lingual_tts",
                data={"text": "Bonjour"},
                files={"reference_audio": ("sample.wav", b"RIFF_DATA", "audio/wav")},
            ).content
            == b"MP3:6:"
        )

    instance.cleanup()
    assert engine.closed is True


def test_health_and_config_endpoints(tmp_path: Path) -> None:
    client, _engine, _handlers, settings = _build_test_client(tmp_path, loaded=True)

    for path in ("/health", "/v1/model/cosyvoice/healthcheck"):
        resp = client.get(path)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["device"] == "cpu"

    cfg = client.get("/config")
    assert cfg.status_code == 200
    data = cfg.json()
    assert data["default_voice"] == "qwen"
    assert data["available_voice_prompts"] == ["qwen"]
    assert data["sample_rate"] == 24000
    assert data["reference_audio_sample_rate"] == 16000
    assert data["model_family"] == "cosyvoice3"
    assert data["model_directory"] == str(settings.model_dir)


def test_unloaded_model_returns_503_for_health_and_synthesis(tmp_path: Path) -> None:
    client, _engine, _handlers, settings = _build_test_client(tmp_path, loaded=False)
    assert client.get("/health").status_code == 503
    assert client.post("/tts", json={"text": "Hi", "stream": False}).status_code == 503
    assert client.post("/tts", json={"text": "Hi", "stream": True}).status_code == 503
    assert (
        client.post(
            "/cross_lingual_tts",
            data={"text": "Hi"},
            files={"reference_audio": ("s.wav", b"RIFF", "audio/wav")},
        ).status_code
        == 503
    )
    assert list(settings.tmp_dir.glob("cross_lingual_*")) == []


def test_tts_standard_buffered_streaming_and_validation(tmp_path: Path) -> None:
    client, engine, _handlers, _settings = _build_test_client(tmp_path)

    resp = client.post("/tts", json={"text": "Hello", "voice_type": "unknown", "speed": 1.2})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert resp.content == b"MP3:6:"
    method, text, stream, kwargs = engine.calls[-1]
    assert method == "inference_cross_lingual"
    assert text == "You are a helpful assistant.<|endofprompt|>Hello"
    assert stream is False
    assert kwargs["speed"] == 1.2

    stream_resp = client.post("/tts", json={"text": "Stream me", "stream": True})
    assert stream_resp.status_code == 200
    assert stream_resp.content == b"MP3:3:MP3:3:"

    assert client.post("/tts", content="not-json").status_code == 400
    assert client.post("/tts", json={"voice_type": "qwen"}).status_code == 400
    assert client.post("/tts", json={"text": "   "}).status_code == 400
    assert client.post("/tts", json={"text": "Hi", "stream": "false"}).status_code == 400
    assert client.post("/tts", json={"text": "Hi", "speed": 3.0}).status_code == 400


def test_instruct_tts_buffered_streaming_and_validation(tmp_path: Path) -> None:
    client, engine, _handlers, _settings = _build_test_client(tmp_path)

    resp = client.post(
        "/instruct_tts",
        json={"text": "Good morning", "instruction": "Speak warmly.", "stream": False},
    )
    assert resp.status_code == 200
    assert resp.content == b"MP3:6:"
    method, text, stream, kwargs = engine.calls[-1]
    assert method == "inference_instruct2"
    assert text == "Good morning"
    assert kwargs["instruct_text"] == "You are a helpful assistant. Speak warmly.<|endofprompt|>"
    assert stream is False

    stream_resp = client.post(
        "/instruct_tts",
        json={"text": "Good morning", "instruction": "Speak warmly.", "stream": True},
    )
    assert stream_resp.status_code == 200
    assert stream_resp.content == b"MP3:3:MP3:3:"

    assert client.post("/instruct_tts", json={"text": "Missing inst"}).status_code == 400
    assert (
        client.post("/instruct_tts", json={"text": "Hi", "instruction": "   "}).status_code == 400
    )
    assert (
        client.post(
            "/instruct_tts",
            json={"text": "Hi", "instruction": "Calm", "stream": "true"},
        ).status_code
        == 400
    )


def test_upload_endpoints_validation_limits_and_cleanup(tmp_path: Path) -> None:
    client, engine, _handlers, settings = _build_test_client(
        tmp_path, env={"MAX_REFERENCE_AUDIO_BYTES": "16"}
    )

    # Non-multipart request must return structured 400 rather than unhandled 500.
    bad_ct = client.post("/zero_shot_tts", json={"text": "Hi"})
    assert bad_ct.status_code == 400
    assert "detail" in bad_ct.json()

    # Empty upload -> 400; oversized upload -> 413; both leave zero temp files.
    empty_resp = client.post(
        "/cross_lingual_tts",
        data={"text": "Hi"},
        files={"reference_audio": ("empty.wav", b"", "audio/wav")},
    )
    assert empty_resp.status_code == 400

    big_resp = client.post(
        "/cross_lingual_tts",
        data={"text": "Hi"},
        files={"reference_audio": ("big.wav", b"0123456789abcdefX", "audio/wav")},
    )
    assert big_resp.status_code == 413
    assert list(settings.tmp_dir.glob("cross_lingual_*")) == []

    for stream_flag in ("false", "true"):
        resp = client.post(
            "/zero_shot_tts",
            data={
                "text": "Clone this",
                "reference_text": "Reference transcript",
                "stream": stream_flag,
            },
            files={"reference_audio": ("../../weird\x00.m3u8", b"RIFF_DATA", "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.content == (b"MP3:3:MP3:3:" if stream_flag == "true" else b"MP3:6:")
        assert list(settings.tmp_dir.glob("zero_shot_*")) == []

    method, text, _stream, kwargs = engine.calls[-1]
    assert method == "inference_zero_shot"
    assert text == "Clone this"
    assert (
        kwargs["prompt_text"] == "You are a helpful assistant.<|endofprompt|>Reference transcript"
    )

    assert client.post("/zero_shot_tts", data={"text": "Only text"}).status_code == 400
    assert (
        client.post(
            "/cross_lingual_tts",
            data={"text": "   "},
            files={"reference_audio": ("a.wav", b"RIFF", "audio/wav")},
        ).status_code
        == 400
    )


def test_corrupt_upload_maps_to_400_without_leaking_internal_paths(tmp_path: Path) -> None:
    async def failing_norm(_src: Path, _dst: Path, _sr: int) -> Path:
        raise AudioProcessingError("ffmpeg stderr: /secret/tmp/raw.audio invalid data")

    client, _engine, _handlers, settings = _build_test_client(tmp_path, ref_normalizer=failing_norm)
    resp = client.post(
        "/cross_lingual_tts",
        data={"text": "Hello"},
        files={"reference_audio": ("bad.wav", b"NOT_AUDIO", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "/secret/" not in resp.json()["detail"]
    assert list(settings.tmp_dir.glob("cross_lingual_*")) == []


def test_mid_stream_error_propagates_and_cleans_temp_file(tmp_path: Path) -> None:
    client, _engine, _handlers, settings = _build_test_client(tmp_path, fail_mid_stream=True)
    with pytest.raises(RuntimeError, match="synthetic mid-stream failure"):
        client.post(
            "/cross_lingual_tts",
            data={"text": "Hello", "stream": "true"},
            files={"reference_audio": ("sample.wav", b"RIFF", "audio/wav")},
        )
    assert list(settings.tmp_dir.glob("cross_lingual_*")) == []


def test_temp_path_guard_cleans_unstarted_stream_on_gc(tmp_path: Path) -> None:
    temp_file = tmp_path / "unstarted.wav"
    temp_file.write_bytes(b"WAV")
    guard = TempPathGuard([temp_file])
    del guard
    gc.collect()
    assert not temp_file.exists()


def test_reconfigure_and_cors_origins(tmp_path: Path) -> None:
    client, _engine, handlers, _settings = _build_test_client(
        tmp_path, env={"CORS_ALLOW_ORIGINS": "https://allowed.example"}
    )
    handlers.voice_prompts["alt"] = tmp_path / "alt.wav"

    asyncio.run(handlers.reconfigure(None))
    asyncio.run(handlers.reconfigure({"default_voice": "nonexistent", "max_replicas": 4}))
    assert handlers.default_voice == "qwen"
    assert handlers.user_config["default_voice"] == "qwen"
    assert handlers.user_config["max_replicas"] == 4

    asyncio.run(handlers.reconfigure({"default_voice": "alt"}))
    assert handlers.default_voice == "alt"
    assert handlers.user_config["default_voice"] == "alt"

    allowed = client.get("/health", headers={"Origin": "https://allowed.example"})
    assert allowed.headers.get("access-control-allow-origin") == "https://allowed.example"
    blocked = client.get("/health", headers={"Origin": "https://evil.example"})
    assert blocked.headers.get("access-control-allow-origin") is None
