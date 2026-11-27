"""Device management, ModelScope download, and CosyVoice3 AutoModel wrapper."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ray
import torch

from cosyvoice_api.audio import normalize_audio_tensor, validate_audio_tensor
from cosyvoice_api.config import Settings

logger = logging.getLogger(__name__)

REQUIRED_MODEL_FILES = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt")


def configure_cuda_environment() -> None:
    """Apply safe CUDA environment defaults without overriding explicit user settings."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")


def configure_torch_cuda() -> None:
    """Enable TF32 and cuDNN autotuning when CUDA is available."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; running on CPU.")
        return

    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    logger.info("PyTorch %s | CUDA devices: %d", torch.__version__, torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        logger.info("CUDA device %d: %s (%.1f GB)", index, props.name, props.total_memory / 1024**3)


def model_files_present(model_dir: Path) -> bool:
    return all((model_dir / filename).exists() for filename in REQUIRED_MODEL_FILES)


class CosyVoiceEngine:
    """Owns the CosyVoice3 model instance, device assignment, and inference execution."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device: str = "cpu"
        self.model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def select_device(self) -> str:
        if not torch.cuda.is_available():
            logger.info("CUDA unavailable; replica using CPU.")
            return "cpu"

        try:
            gpu_ids = ray.get_gpu_ids()
            if not gpu_ids:
                logger.warning("CUDA is available, but Ray assigned no GPU; using CPU.")
                return "cpu"

            gpu_id = int(gpu_ids[0])
            device_str = f"cuda:{gpu_id}"
            _ = torch.tensor([1.0], device=device_str)

            props = torch.cuda.get_device_properties(gpu_id)
            allocated_mb = torch.cuda.memory_allocated(device_str) / 1024**2
            logger.info(
                "Replica assigned %s (%s, %.1f GB total, %.1f MB allocated)",
                device_str,
                props.name,
                props.total_memory / 1024**3,
                allocated_mb,
            )
            return device_str
        except Exception as exc:
            logger.error("GPU setup failed (%s); falling back to CPU.", exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "cpu"

    def load(self) -> None:
        """Download weights if absent, then initialize CosyVoice3 with CPU fallback."""
        configure_cuda_environment()
        configure_torch_cuda()
        self.settings.ensure_runtime_dirs()
        self.device = self.select_device()

        local_model_dir = self.settings.model_dir
        if not model_files_present(local_model_dir):
            from modelscope import snapshot_download

            logger.info(
                "Downloading %s (revision=%s) to %s",
                self.settings.model_id,
                self.settings.model_revision,
                local_model_dir,
            )
            snapshot_download(
                self.settings.model_id,
                local_dir=str(local_model_dir),
                revision=self.settings.model_revision,
            )
        else:
            logger.info("Using local model files at %s", local_model_dir)

        # Lazy import avoids pulling vendored cosyvoice logging side effects on module import.
        from cosyvoice.cli.cosyvoice import AutoModel

        use_cuda = self.device.startswith("cuda")
        try:
            self.model = AutoModel(
                model_dir=str(local_model_dir),
                load_trt=False,
                load_vllm=False,
                fp16=use_cuda,
            )
            logger.info("Loaded CosyVoice3 model on %s", self.device)
            if use_cuda and torch.cuda.is_available():
                logger.info(
                    "GPU memory on %s: allocated=%.1f MB, reserved=%.1f MB",
                    self.device,
                    torch.cuda.memory_allocated(self.device) / 1024**2,
                    torch.cuda.memory_reserved(self.device) / 1024**2,
                )
        except Exception as exc:
            logger.error("Model load failed on %s: %s", self.device, exc, exc_info=True)
            if not use_cuda:
                raise RuntimeError(f"CosyVoice model could not be loaded: {exc}") from exc

            logger.warning("Retrying model initialization on CPU without fp16.")
            self.device = "cpu"
            try:
                self.model = AutoModel(
                    model_dir=str(local_model_dir),
                    load_trt=False,
                    load_vllm=False,
                    fp16=False,
                )
                logger.info("Loaded CosyVoice3 model on CPU fallback.")
            except Exception as cpu_exc:
                logger.error("CPU fallback model load failed: %s", cpu_exc, exc_info=True)
                raise RuntimeError(
                    "CosyVoice model could not be loaded on GPU or CPU."
                ) from cpu_exc

    def iter_speech_chunks(
        self,
        method_name: str,
        text: str,
        *,
        stream: bool,
        **kwargs: Any,
    ) -> Iterator[torch.Tensor]:
        if self.model is None:
            raise RuntimeError("CosyVoice model is not loaded.")

        inference_fn = getattr(self.model, method_name, None)
        if not callable(inference_fn):
            raise AttributeError(f"Inference method '{method_name}' not found on model.")

        logger.info("Synthesizing via %s (stream=%s, chars=%d)", method_name, stream, len(text))
        for idx, result in enumerate(inference_fn(text, **kwargs, stream=stream)):
            chunk = result.get("tts_speech") if isinstance(result, dict) else None
            if chunk is None:
                continue
            if not validate_audio_tensor(chunk):
                logger.warning("Skipping invalid audio chunk %d from %s", idx, method_name)
                continue
            yield normalize_audio_tensor(chunk)

    def synthesize(self, method_name: str, text: str, **kwargs: Any) -> torch.Tensor:
        chunks = [
            chunk.cpu()
            for chunk in self.iter_speech_chunks(method_name, text, stream=False, **kwargs)
        ]
        if not chunks:
            raise RuntimeError("TTS generation failed to produce valid audio.")

        try:
            combined = torch.cat(chunks, dim=-1)
        except Exception as exc:
            shapes = [tuple(chunk.shape) for chunk in chunks]
            logger.error("Failed to concatenate chunks %s: %s", shapes, exc)
            raise RuntimeError("Failed to combine audio chunks.") from exc

        logger.info(
            "Synthesized %d chunk(s) via %s (%.2fs)",
            len(chunks),
            method_name,
            combined.shape[-1] / self.settings.sample_rate,
        )
        return combined

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Released CosyVoice model.")
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                logger.error("Failed to clear CUDA cache on %s: %s", self.device, exc)
