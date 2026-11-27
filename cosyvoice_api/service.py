"""HTTP route handlers and per-replica resource lifecycle for CosyVoice3."""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from cosyvoice_api.audio import (
    AudioProcessingError,
    VoicePromptCache,
    encode_tensor_to_mp3_bytes_async,
    normalize_reference_audio_async,
    safe_unlink,
)
from cosyvoice_api.config import Settings, configure_logging
from cosyvoice_api.engine import CosyVoiceEngine
from cosyvoice_api.http_utils import (
    TempPathGuard,
    configure_cors,
    extract_upload_fields,
    get_ray_context_ids,
    parse_json_stream_flag,
    parse_speed_or_400,
    read_form_or_400,
    read_json_object,
    stream_upload_to_file,
)
from cosyvoice_api.prompts import format_instruction, format_prompt_text

logger = logging.getLogger(__name__)

Mp3Encoder = Callable[[torch.Tensor, Path, int], Coroutine[Any, Any, bytes]]
RefNormalizer = Callable[[Path, Path, int], Coroutine[Any, Any, Path]]


class CosyVoiceHandlers:
    """Framework-agnostic HTTP handlers and model/cache lifecycle."""

    def __init__(
        self,
        user_config: dict[str, Any] | None = None,
        *,
        settings: Settings | None = None,
        engine: Any | None = None,
        mp3_encoder: Mp3Encoder = encode_tensor_to_mp3_bytes_async,
        ref_normalizer: RefNormalizer = normalize_reference_audio_async,
        auto_load: bool = True,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_runtime_dirs()
        configure_logging(self.settings)

        self.user_config: dict[str, Any] = dict(user_config or {})
        self.voice_prompts: dict[str, Path] = dict(self.settings.voice_prompts)
        self.default_voice: str = str(
            self.user_config.get("default_voice", self.settings.default_voice)
        )
        self._mp3_encoder = mp3_encoder
        self._ref_normalizer = ref_normalizer
        self._voice_cache = VoicePromptCache(
            tmp_dir=self.settings.tmp_dir,
            sample_rate=self.settings.reference_audio_sample_rate,
            normalizer=ref_normalizer,
        )
        # Vendored CosyVoice3Model holds a single non-reentrant CUDA stream context;
        # serialize model execution per replica while keeping the event loop responsive.
        self._inference_lock = asyncio.Lock()

        self._log_voice_prompts()

        if engine is not None:
            self.engine = engine
        else:
            self.engine = CosyVoiceEngine(self.settings)
            if auto_load:
                self.engine.load()

    def _log_voice_prompts(self) -> None:
        for key, path in self.voice_prompts.items():
            if path.exists():
                logger.info("Built-in voice '%s' registered at %s", key, path)
            else:
                logger.error("Built-in voice '%s' missing at %s", key, path)
        if self.default_voice not in self.voice_prompts:
            logger.warning("Default voice '%s' is not in VOICE_PROMPTS.", self.default_voice)

    def cleanup(self) -> None:
        try:
            self.engine.close()
        finally:
            self._voice_cache.clear()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def _require_model_loaded(self) -> None:
        if not self.engine.is_loaded:
            raise HTTPException(status_code=503, detail="CosyVoice model is not loaded.")

    def _resolve_voice(self, requested_voice: Any) -> str:
        voice = str(requested_voice) if requested_voice else self.default_voice
        if voice in self.voice_prompts:
            return voice

        logger.warning("Voice '%s' not found; falling back to '%s'.", voice, self.default_voice)
        if self.default_voice not in self.voice_prompts:
            raise HTTPException(
                status_code=400,
                detail=f"Default voice prompt '{self.default_voice}' is not available.",
            )
        return self.default_voice

    async def _prepare_uploaded_reference(self, upload: UploadFile, prefix: str) -> Path:
        token = uuid.uuid4().hex
        raw_path = self.settings.tmp_dir / f"{prefix}_raw_{token}.audio"
        normalized_path = self.settings.tmp_dir / f"{prefix}_norm_{token}.wav"

        try:
            await stream_upload_to_file(
                upload, raw_path, max_bytes=self.settings.max_reference_audio_bytes
            )
            await self._ref_normalizer(
                raw_path,
                normalized_path,
                self.settings.reference_audio_sample_rate,
            )
            return normalized_path
        except HTTPException:
            safe_unlink(normalized_path)
            raise
        except AudioProcessingError as exc:
            safe_unlink(normalized_path)
            logger.warning("Invalid uploaded reference audio (%s): %s", prefix, exc)
            raise HTTPException(
                status_code=400,
                detail="Invalid or unsupported reference audio file.",
            ) from None
        except Exception:
            safe_unlink(normalized_path)
            raise
        finally:
            safe_unlink(raw_path)

    async def _build_audio_response(
        self,
        *,
        method_name: str,
        text: str,
        stream: bool,
        cleanup_paths: list[Path] | None = None,
        **inference_kwargs: Any,
    ) -> Response:
        guard = TempPathGuard(cleanup_paths or [])
        try:
            self._require_model_loaded()

            if stream:
                return StreamingResponse(
                    self._stream_mp3(
                        method_name=method_name,
                        text=text,
                        guard=guard,
                        **inference_kwargs,
                    ),
                    media_type="audio/mpeg",
                )

            async with self._inference_lock:
                audio_tensor = await asyncio.to_thread(
                    self.engine.synthesize,
                    method_name,
                    text,
                    **inference_kwargs,
                )
            mp3_bytes = await self._mp3_encoder(
                audio_tensor,
                self.settings.tmp_dir,
                self.settings.sample_rate,
            )
            guard.release()
            return Response(mp3_bytes, media_type="audio/mpeg")
        except BaseException:
            guard.release()
            raise

    async def _stream_mp3(
        self,
        *,
        method_name: str,
        text: str,
        guard: TempPathGuard,
        **inference_kwargs: Any,
    ) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stop_event = threading.Event()

        def _produce() -> None:
            try:
                for chunk in self.engine.iter_speech_chunks(
                    method_name, text, stream=True, **inference_kwargs
                ):
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        chunk_count = 0
        try:
            async with self._inference_lock:
                producer_task = asyncio.create_task(asyncio.to_thread(_produce))
                try:
                    while True:
                        kind, item = await queue.get()
                        if kind == "done":
                            break
                        if kind == "error":
                            raise item
                        yield await self._mp3_encoder(
                            item,
                            self.settings.tmp_dir,
                            self.settings.sample_rate,
                        )
                        chunk_count += 1
                    if chunk_count == 0:
                        logger.warning("Streaming %s produced no audio chunks.", method_name)
                except Exception as exc:
                    logger.error("Streaming %s failed: %s", method_name, exc, exc_info=True)
                    raise
                finally:
                    stop_event.set()
                    await asyncio.shield(producer_task)
        finally:
            guard.release()

    async def _run_uploaded_synthesis(
        self,
        *,
        upload: UploadFile,
        prefix: str,
        method_name: str,
        text: str,
        stream: bool,
        error_label: str,
        **inference_kwargs: Any,
    ) -> Response:
        self._require_model_loaded()
        normalized_ref: Path | None = None
        try:
            normalized_ref = await self._prepare_uploaded_reference(upload, prefix)
            response = await self._build_audio_response(
                method_name=method_name,
                text=text,
                stream=stream,
                cleanup_paths=[normalized_ref],
                prompt_wav=str(normalized_ref),
                **inference_kwargs,
            )
            normalized_ref = None
            return response
        except HTTPException:
            if normalized_ref is not None:
                safe_unlink(normalized_ref)
            raise
        except FileNotFoundError as exc:
            if normalized_ref is not None:
                safe_unlink(normalized_ref)
            raise HTTPException(status_code=400, detail="Reference audio file not found.") from exc
        except Exception as exc:
            if normalized_ref is not None:
                safe_unlink(normalized_ref)
            logger.error("%s failed: %s", error_label, exc, exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"{error_label} failed due to an internal error."
            ) from exc

    async def health_check(self, request: Request) -> JSONResponse:
        if not self.engine.is_loaded:
            return JSONResponse(
                {"status": "unhealthy", "reason": "Model not loaded"},
                status_code=503,
            )

        actor_id, node_id = get_ray_context_ids()
        return JSONResponse(
            {
                "status": "healthy",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "device": self.engine.device,
                "model_loaded": True,
                "ray_actor_id": actor_id,
                "ray_node_id": node_id,
            }
        )

    async def get_config(self, request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "default_voice": self.default_voice,
                "available_voice_prompts": list(self.voice_prompts.keys()),
                "sample_rate": self.settings.sample_rate,
                "reference_audio_sample_rate": self.settings.reference_audio_sample_rate,
                "device_in_use": self.engine.device,
                "model_directory": str(self.settings.model_dir),
                "model_id": self.settings.model_id,
                "model_dir_name": self.settings.model_dir_name,
                "model_revision": self.settings.model_revision,
                "model_family": self.settings.model_family,
                "max_replicas_config": self.user_config.get(
                    "max_replicas", self.settings.max_replicas
                ),
            }
        )

    async def tts_standard(self, request: Request) -> Response:
        payload = await read_json_object(request)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=400, detail="Missing or invalid 'text' field in payload."
            )

        voice_type = self._resolve_voice(payload.get("voice_type"))
        stream = parse_json_stream_flag(payload.get("stream"))
        speed = parse_speed_or_400(payload.get("speed", 1.0))
        self._require_model_loaded()

        try:
            prompt_wav = await self._voice_cache.get_or_prepare(
                voice_type, self.voice_prompts[voice_type]
            )
            return await self._build_audio_response(
                method_name="inference_cross_lingual",
                text=format_prompt_text(text),
                stream=stream,
                prompt_wav=str(prompt_wav),
                speed=speed,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("POST /tts failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500, detail="TTS generation failed due to an internal error."
            ) from exc

    async def tts_zero_shot(self, request: Request) -> Response:
        form = await read_form_or_400(request)
        try:
            text, reference_text, reference_audio, stream, speed = extract_upload_fields(
                form, require_reference_text=True
            )
            assert reference_text is not None
            return await self._run_uploaded_synthesis(
                upload=reference_audio,
                prefix="zero_shot",
                method_name="inference_zero_shot",
                text=text,
                stream=stream,
                error_label="Zero-shot TTS generation",
                prompt_text=format_prompt_text(reference_text),
                speed=speed,
            )
        finally:
            await form.close()

    async def tts_cross_lingual(self, request: Request) -> Response:
        form = await read_form_or_400(request)
        try:
            text, _, reference_audio, stream, speed = extract_upload_fields(
                form, require_reference_text=False
            )
            return await self._run_uploaded_synthesis(
                upload=reference_audio,
                prefix="cross_lingual",
                method_name="inference_cross_lingual",
                text=format_prompt_text(text),
                stream=stream,
                error_label="Cross-lingual TTS generation",
                speed=speed,
            )
        finally:
            await form.close()

    async def tts_instruct(self, request: Request) -> Response:
        payload = await read_json_object(request)
        text = payload.get("text")
        instruction = payload.get("instruction")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=400, detail="Missing or invalid 'text' field.")
        if not isinstance(instruction, str) or not instruction.strip():
            raise HTTPException(status_code=400, detail="Missing or invalid 'instruction' field.")

        voice_type = self._resolve_voice(payload.get("voice_type"))
        stream = parse_json_stream_flag(payload.get("stream"))
        speed = parse_speed_or_400(payload.get("speed", 1.0))
        self._require_model_loaded()

        try:
            prompt_wav = await self._voice_cache.get_or_prepare(
                voice_type, self.voice_prompts[voice_type]
            )
            return await self._build_audio_response(
                method_name="inference_instruct2",
                text=text.strip(),
                stream=stream,
                instruct_text=format_instruction(instruction),
                prompt_wav=str(prompt_wav),
                speed=speed,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("POST /instruct_tts failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500, detail="Instruct TTS generation failed due to an internal error."
            ) from exc

    async def reconfigure(self, config: dict[str, Any] | None) -> None:
        if not config:
            return
        logger.info("Applying runtime user_config update: %s", config)
        sanitized = dict(config)
        candidate_voice = sanitized.get("default_voice")
        if candidate_voice is not None:
            if candidate_voice in self.voice_prompts:
                self.default_voice = candidate_voice
            else:
                logger.warning(
                    "Ignoring unknown default_voice '%s' in reconfigure; keeping '%s'.",
                    candidate_voice,
                    self.default_voice,
                )
                sanitized["default_voice"] = self.default_voice
        self.user_config.update(sanitized)


def create_asgi_app(handlers: CosyVoiceHandlers) -> FastAPI:
    """Create a standalone FastAPI app bound to a handler instance (for local ASGI/tests)."""
    standalone = FastAPI(title="CosyVoice Ray Serve API")
    configure_cors(standalone, handlers.settings.cors_allow_origins)
    standalone.add_api_route("/health", handlers.health_check, methods=["GET"])
    standalone.add_api_route(
        "/v1/model/cosyvoice/healthcheck", handlers.health_check, methods=["GET"]
    )
    standalone.add_api_route("/config", handlers.get_config, methods=["GET"])
    standalone.add_api_route("/tts", handlers.tts_standard, methods=["POST"])
    standalone.add_api_route("/zero_shot_tts", handlers.tts_zero_shot, methods=["POST"])
    standalone.add_api_route("/cross_lingual_tts", handlers.tts_cross_lingual, methods=["POST"])
    standalone.add_api_route("/instruct_tts", handlers.tts_instruct, methods=["POST"])
    return standalone
