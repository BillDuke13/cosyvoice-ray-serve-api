"""Device management, locked ModelScope download, and CosyVoice3 AutoModel wrapper."""

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ray
import torch
from filelock import FileLock

from cosyvoice_api.audio import normalize_audio_tensor, validate_audio_tensor
from cosyvoice_api.config import Settings

logger = logging.getLogger(__name__)

REQUIRED_MODEL_FILES = (
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/config.json",
)


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
    return all((model_dir / rel_path).exists() for rel_path in REQUIRED_MODEL_FILES)


def resolve_local_cuda_ordinal(assigned_gpu_id: Any, device_count: int) -> int:
    """Map a Ray-assigned physical GPU ID or UUID to this process's local CUDA ordinal.

    Ray isolates GPU actors via CUDA_VISIBLE_DEVICES, so an actor assigned physical GPU 1
    typically sees only 1 visible CUDA device at ordinal 0.
    """
    if device_count <= 1:
        return 0

    assigned_str = str(assigned_gpu_id).strip()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
        if assigned_str in visible_ids:
            idx = visible_ids.index(assigned_str)
            if idx < device_count:
                return idx

    try:
        numeric_id = int(assigned_str)
    except ValueError:
        return 0
    if 0 <= numeric_id < device_count:
        return numeric_id
    return 0


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

            device_count = torch.cuda.device_count()
            local_ordinal = resolve_local_cuda_ordinal(gpu_ids[0], device_count)
            device_str = f"cuda:{local_ordinal}"
            _ = torch.tensor([1.0], device=device_str)

            props = torch.cuda.get_device_properties(local_ordinal)
            allocated_mb = torch.cuda.memory_allocated(device_str) / 1024**2
            logger.info(
                "Replica assigned %s (ray_gpu=%s, %s, %.1f GB total, %.1f MB allocated)",
                device_str,
                gpu_ids[0],
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

    def _ensure_model_downloaded(self) -> Path:
        local_model_dir = self.settings.model_dir
        if model_files_present(local_model_dir):
            logger.info("Using local model files at %s", local_model_dir)
            return local_model_dir

        lock_path = local_model_dir.parent / f".{local_model_dir.name}.download.lock"
        with FileLock(str(lock_path)):
            if model_files_present(local_model_dir):
                logger.info("Using local model files at %s", local_model_dir)
                return local_model_dir

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
        return local_model_dir

    def _init_automodel(self, local_model_dir: Path) -> Any:
        # Lazy import avoids triggering vendored cosyvoice logging side effects at import time.
        from cosyvoice.cli.cosyvoice import AutoModel

        use_cuda = self.device.startswith("cuda")
        try:
            model = AutoModel(
                model_dir=str(local_model_dir),
                load_trt=False,
                load_vllm=False,
                fp16=use_cuda,
            )
            logger.info("Loaded CosyVoice3 model on %s (fp16=%s)", self.device, use_cuda)
            if use_cuda and torch.cuda.is_available():
                logger.info(
                    "GPU memory on %s: allocated=%.1f MB, reserved=%.1f MB",
                    self.device,
                    torch.cuda.memory_allocated(self.device) / 1024**2,
                    torch.cuda.memory_reserved(self.device) / 1024**2,
                )
            return model
        except Exception as exc:
            logger.error("Model load failed on %s: %s", self.device, exc, exc_info=True)
            if not use_cuda:
                raise RuntimeError(f"CosyVoice model could not be loaded: {exc}") from exc

            # Vendored CosyVoice3 selects CUDA whenever torch.cuda.is_available() is True,
            # so retry without fp16 rather than claiming CPU execution on a CUDA-enabled process.
            logger.warning("Retrying CosyVoice3 initialization with fp16=False.")
            try:
                model = AutoModel(
                    model_dir=str(local_model_dir),
                    load_trt=False,
                    load_vllm=False,
                    fp16=False,
                )
                logger.info("Loaded CosyVoice3 model on %s with fp16=False.", self.device)
                return model
            except Exception as retry_exc:
                logger.error("Retry with fp16=False failed: %s", retry_exc, exc_info=True)
                raise RuntimeError(
                    "CosyVoice model could not be loaded (fp16=True and fp16=False both failed)."
                ) from retry_exc

    def load(self) -> None:
        """Download weights if absent, then initialize CosyVoice3."""
        configure_cuda_environment()
        configure_torch_cuda()
        self.settings.ensure_runtime_dirs()
        self.device = self.select_device()
        local_model_dir = self._ensure_model_downloaded()
        self.model = self._init_automodel(local_model_dir)

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
