from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cosyvoice_api.audio import (
    AudioProcessingError,
    VoicePromptCache,
    encode_tensor_to_mp3_bytes,
    normalize_audio_tensor,
    normalize_reference_audio,
    validate_audio_tensor,
)


def test_validate_audio_tensor() -> None:
    assert validate_audio_tensor(torch.tensor([0.1, -0.2, 0.3])) is True
    assert validate_audio_tensor("not-a-tensor") is False
    assert validate_audio_tensor(torch.empty(0)) is False
    assert validate_audio_tensor(torch.tensor([0.1, float("nan")])) is False
    assert validate_audio_tensor(torch.tensor([0.1, float("inf")])) is False


def test_normalize_audio_tensor_1d_and_2d() -> None:
    mono = normalize_audio_tensor(torch.tensor([0.25, -0.5, 0.125]))
    assert mono.dtype == torch.float32
    assert torch.isclose(mono.abs().max(), torch.tensor(1.0))

    stereo = normalize_audio_tensor(torch.tensor([[0.2, -0.4], [0.0, 0.0]], dtype=torch.float64))
    assert stereo.dtype == torch.float32
    assert torch.isclose(stereo[0].abs().max(), torch.tensor(1.0))
    assert torch.equal(stereo[1], torch.zeros(2))


def test_normalize_reference_audio_hardened_flags_and_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        normalize_reference_audio(tmp_path / "missing.wav", tmp_path / "out.wav")

    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    src.write_bytes(b"RIFF")
    captured: dict[str, object] = {}

    def fake_ok(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        dst.write_bytes(b"RIFF_OUT")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("cosyvoice_api.audio.subprocess.run", side_effect=fake_ok):
        assert normalize_reference_audio(src, dst) == dst

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "-nostdin" in cmd
    assert "-protocol_whitelist" in cmd
    assert "file,pipe" in cmd
    assert "-vn" in cmd and "-sn" in cmd and "-dn" in cmd
    assert "-t" in cmd and "30" in cmd
    assert captured["timeout"] == 60

    def fake_fail(cmd: list[str], **_kwargs: object) -> None:
        dst.write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, cmd, stderr="bad input")

    with (
        patch("cosyvoice_api.audio.subprocess.run", side_effect=fake_fail),
        pytest.raises(AudioProcessingError, match="bad input"),
    ):
        normalize_reference_audio(src, dst)
    assert not dst.exists()


def test_encode_tensor_to_mp3_bytes_cleans_temp_files_and_errors(tmp_path: Path) -> None:
    tensor = torch.linspace(-0.5, 0.5, steps=2400).unsqueeze(0)
    fake_mp3 = b"ID3" + (b"\x00" * 600)

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(cmd[cmd.index("-qscale:a") + 2])
        output_path.write_bytes(fake_mp3)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("cosyvoice_api.audio.subprocess.run", side_effect=fake_run):
        result = encode_tensor_to_mp3_bytes(tensor, tmp_path, sample_rate=24000)

    assert result == fake_mp3
    assert list(tmp_path.iterdir()) == []

    # Missing ffmpeg must report ffmpeg execution failure, not WAV serialization failure.
    with (
        patch("cosyvoice_api.audio.subprocess.run", side_effect=FileNotFoundError("ffmpeg")),
        pytest.raises(AudioProcessingError, match="Unable to execute ffmpeg"),
    ):
        encode_tensor_to_mp3_bytes(tensor, tmp_path)
    assert list(tmp_path.iterdir()) == []

    def fake_empty(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(cmd[cmd.index("-qscale:a") + 2])
        output_path.write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with (
        patch("cosyvoice_api.audio.subprocess.run", side_effect=fake_empty),
        pytest.raises(AudioProcessingError, match="empty MP3"),
    ):
        encode_tensor_to_mp3_bytes(tensor, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_voice_prompt_cache_concurrent_stampede_retirement_and_cancel(tmp_path: Path) -> None:
    source = tmp_path / "qwen.wav"
    source.write_bytes(b"v1")
    cache_dir = tmp_path / "cache"
    calls = 0

    async def slow_norm(_src: Path, dst: Path, _sr: int) -> Path:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        dst.write_bytes(f"norm-{calls}".encode())
        return dst

    async def _run() -> None:
        cache = VoicePromptCache(tmp_dir=cache_dir, normalizer=slow_norm)
        # Concurrent requests for the same voice must serialize and normalize only once.
        paths = await asyncio.gather(
            cache.get_or_prepare("qwen", source),
            cache.get_or_prepare("qwen", source),
            cache.get_or_prepare("qwen", source),
        )
        assert len(set(paths)) == 1
        assert calls == 1
        first = paths[0]

        # On mtime change, old file stays alive (retired) for in-flight readers until clear().
        new_mtime = source.stat().st_mtime + 5.0
        os.utime(source, (new_mtime, new_mtime))
        second = await cache.get_or_prepare("qwen", source)
        assert second != first
        assert calls == 2
        assert first.exists()
        assert second.exists()

        cache.clear()
        assert not first.exists()
        assert not second.exists()

        # Cancellation mid-normalization must unlink any partially written output.
        async def failing_norm(_src: Path, dst: Path, _sr: int) -> Path:
            dst.write_bytes(b"partial")
            raise asyncio.CancelledError()

        cancel_cache = VoicePromptCache(tmp_dir=cache_dir, normalizer=failing_norm)
        with pytest.raises(asyncio.CancelledError):
            await cancel_cache.get_or_prepare("qwen", source)
        assert list(cache_dir.glob("voice_qwen_*.wav")) == []

    asyncio.run(_run())
