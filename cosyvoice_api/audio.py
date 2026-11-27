"""Audio tensor validation, peak normalization, FFmpeg conversion, and prompt caching."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from cosyvoice_api.config import REFERENCE_AUDIO_SAMPLE_RATE, SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioProcessingError(RuntimeError):
    """Raised when audio normalization, WAV serialization, or MP3 encoding fails."""


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to delete temporary file %s: %s", path, exc)


def validate_audio_tensor(audio_tensor: Any) -> bool:
    if not isinstance(audio_tensor, torch.Tensor):
        logger.warning("Invalid audio output: expected torch.Tensor, got %s", type(audio_tensor))
        return False
    if audio_tensor.numel() == 0:
        logger.warning("Invalid audio output: empty tensor")
        return False
    if torch.isnan(audio_tensor).any():
        logger.warning("Invalid audio output: contains NaN")
        return False
    if torch.isinf(audio_tensor).any():
        logger.warning("Invalid audio output: contains Inf")
        return False
    return True


def normalize_audio_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Peak-normalize audio samples to [-1, 1] without amplifying silence."""
    tensor = audio_tensor.float() if audio_tensor.dtype != torch.float32 else audio_tensor

    if tensor.dim() > 1:
        max_vals = tensor.abs().max(dim=-1, keepdim=True)[0]
        return torch.where(max_vals > 1e-6, tensor / max_vals, tensor)

    max_val = tensor.abs().max()
    if max_val > 1e-6:
        return tensor / max_val
    return tensor


def normalize_reference_audio(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int = REFERENCE_AUDIO_SAMPLE_RATE,
) -> Path:
    """Convert an arbitrary audio clip to a mono PCM-16 WAV at target_sample_rate via FFmpeg."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(target_sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(output_path),
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        safe_unlink(output_path)
        detail = (exc.stderr or "").strip() or str(exc)
        raise AudioProcessingError(f"FFmpeg reference normalization failed: {detail}") from exc
    except OSError as exc:
        safe_unlink(output_path)
        raise AudioProcessingError(f"Unable to execute ffmpeg: {exc}") from exc

    return output_path


def encode_tensor_to_mp3_bytes(
    audio_tensor: torch.Tensor,
    tmp_dir: Path,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Serialize a float tensor to PCM-16 WAV via soundfile, then encode to LAME VBR MP3."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temp_wav = tmp_dir / f"synth_{token}.wav"
    temp_mp3 = tmp_dir / f"synth_{token}.mp3"

    try:
        # soundfile expects (frames,) or (frames, channels), transposed from torch's
        # (channels, frames). Using soundfile avoids torchaudio>=2.9 TorchCodec ignoring
        # PCM encoding options.
        audio_np = audio_tensor.detach().cpu().float().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.T
        sf.write(str(temp_wav), audio_np, sample_rate, subtype="PCM_16")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(temp_wav),
                "-codec:a",
                "libmp3lame",
                "-qscale:a",
                "2",
                str(temp_mp3),
                "-hide_banner",
                "-loglevel",
                "error",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if not temp_mp3.exists() or temp_mp3.stat().st_size == 0:
            raise AudioProcessingError("FFmpeg produced an empty MP3 file.")

        audio_bytes = temp_mp3.read_bytes()
        if len(audio_bytes) < 512:
            logger.warning("Encoded MP3 output is unusually small (%d bytes).", len(audio_bytes))
        return audio_bytes
    except AudioProcessingError:
        raise
    except (RuntimeError, OSError) as exc:
        raise AudioProcessingError(f"WAV serialization failed: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or str(exc)
        raise AudioProcessingError(f"FFmpeg MP3 encoding failed: {detail}") from exc
    finally:
        safe_unlink(temp_wav)
        safe_unlink(temp_mp3)


async def normalize_reference_audio_async(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int = REFERENCE_AUDIO_SAMPLE_RATE,
) -> Path:
    return await asyncio.to_thread(
        normalize_reference_audio,
        input_path,
        output_path,
        target_sample_rate,
    )


async def encode_tensor_to_mp3_bytes_async(
    audio_tensor: torch.Tensor,
    tmp_dir: Path,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    return await asyncio.to_thread(
        encode_tensor_to_mp3_bytes,
        audio_tensor,
        tmp_dir,
        sample_rate,
    )


class VoicePromptCache:
    """Caches normalized 16 kHz mono WAV paths for static built-in voice prompts."""

    def __init__(
        self,
        tmp_dir: Path,
        sample_rate: int = REFERENCE_AUDIO_SAMPLE_RATE,
        *,
        normalizer: Callable[
            [Path, Path, int], Coroutine[Any, Any, Path]
        ] = normalize_reference_audio_async,
    ) -> None:
        self._tmp_dir = tmp_dir
        self._sample_rate = sample_rate
        self._normalizer = normalizer
        self._entries: dict[str, tuple[float, Path]] = {}

    async def get_or_prepare(self, voice_type: str, source_path: Path) -> Path:
        if not source_path.exists():
            raise FileNotFoundError(f"Voice prompt file not found: {source_path}")

        mtime = source_path.stat().st_mtime
        cached = self._entries.get(voice_type)
        if cached is not None:
            cached_mtime, cached_path = cached
            if cached_mtime == mtime and cached_path.exists():
                return cached_path
            safe_unlink(cached_path)
            self._entries.pop(voice_type, None)

        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._tmp_dir / f"voice_{voice_type}_{uuid.uuid4().hex}.wav"
        await self._normalizer(source_path, output_path, self._sample_rate)
        self._entries[voice_type] = (mtime, output_path)
        logger.info("Cached normalized built-in voice '%s' at %s", voice_type, output_path)
        return output_path

    def clear(self) -> None:
        for _, cached_path in self._entries.values():
            safe_unlink(cached_path)
        self._entries.clear()
