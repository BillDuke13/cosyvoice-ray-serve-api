from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from fastapi.testclient import TestClient

from cosyvoice_api import app as ray_app
from cosyvoice_api import http_app
from cosyvoice_api.config import Settings
from cosyvoice_api.service import CosyVoiceHandlers, create_asgi_app


class FakeEngine:
    def __init__(self, *, loaded: bool = True, device: str = "cpu") -> None:
        self._loaded = loaded
        self.device = device
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
    tmp_path: Path, *, loaded: bool = True
) -> tuple[TestClient, FakeEngine, CosyVoiceHandlers, Settings]:
    settings = Settings.from_env({}, root_dir=tmp_path)
    settings.ensure_runtime_dirs()
    settings.voice_prompts["qwen"].write_bytes(b"RIFF_QWEN")
    engine = FakeEngine(loaded=loaded)
    handlers = CosyVoiceHandlers(
        settings=settings,
        engine=engine,
        mp3_encoder=_fake_mp3_encoder,
        ref_normalizer=_fake_ref_normalizer,
    )
    client = TestClient(create_asgi_app(handlers))
    return client, engine, handlers, settings


def test_ray_serve_ingress_routes_and_binding() -> None:
    assert ray_app is not None
    schema_paths = http_app.openapi()["paths"]
    routes = {
        (method.upper(), path) for path, operations in schema_paths.items() for method in operations
    }
    expected = {
        ("GET", "/health"),
        ("GET", "/v1/model/cosyvoice/healthcheck"),
        ("GET", "/config"),
        ("POST", "/tts"),
        ("POST", "/zero_shot_tts"),
        ("POST", "/cross_lingual_tts"),
        ("POST", "/instruct_tts"),
    }
    assert expected.issubset(routes)


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


def test_health_unloaded_returns_503(tmp_path: Path) -> None:
    client, _engine, _handlers, _settings = _build_test_client(tmp_path, loaded=False)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"


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


def test_zero_shot_and_cross_lingual_cleanup_uploaded_temps(tmp_path: Path) -> None:
    client, engine, _handlers, settings = _build_test_client(tmp_path)

    for stream_flag in ("false", "true"):
        resp = client.post(
            "/zero_shot_tts",
            data={
                "text": "Clone this",
                "reference_text": "Reference transcript",
                "stream": stream_flag,
            },
            files={"reference_audio": ("sample.wav", b"RIFF_DATA", "audio/wav")},
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

    for stream_flag in ("false", "true"):
        resp = client.post(
            "/cross_lingual_tts",
            data={"text": "Bonjour", "stream": stream_flag},
            files={"reference_audio": ("sample.wav", b"RIFF_DATA", "audio/wav")},
        )
        assert resp.status_code == 200
        assert resp.content == (b"MP3:3:MP3:3:" if stream_flag == "true" else b"MP3:6:")
        assert list(settings.tmp_dir.glob("cross_lingual_*")) == []

    assert client.post("/zero_shot_tts", data={"text": "Only text"}).status_code == 400
    assert client.post("/cross_lingual_tts", data={"text": "Only text"}).status_code == 400


def test_reconfigure_validates_default_voice(tmp_path: Path) -> None:
    _client, _engine, handlers, _settings = _build_test_client(tmp_path)
    handlers.voice_prompts["alt"] = tmp_path / "alt.wav"

    asyncio.run(handlers.reconfigure({"default_voice": "nonexistent"}))
    assert handlers.default_voice == "qwen"

    asyncio.run(handlers.reconfigure({"default_voice": "alt"}))
    assert handlers.default_voice == "alt"
