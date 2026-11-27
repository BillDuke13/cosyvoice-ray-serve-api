"""HTTP request parsing, upload streaming, CORS, and temp-file cleanup helpers."""

import asyncio
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ray
from fastapi import FastAPI
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from cosyvoice_api.audio import safe_unlink
from cosyvoice_api.config import DEFAULT_CORS_ALLOW_ORIGINS
from cosyvoice_api.prompts import parse_speed

UPLOAD_READ_CHUNK_BYTES = 64 * 1024


def configure_cors(
    app: FastAPI,
    allow_origins: Sequence[str] = DEFAULT_CORS_ALLOW_ORIGINS,
) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allow_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _unlink_paths(paths: list[Path]) -> None:
    for path in paths:
        safe_unlink(path)


class TempPathGuard:
    """Deletes request-scoped temp files on release or GC (e.g. unstarted StreamingResponse)."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = list(paths)
        self._finalizer = weakref.finalize(self, _unlink_paths, self._paths)

    def release(self) -> None:
        if self._finalizer.alive:
            self._finalizer()


def parse_speed_or_400(raw_speed: Any) -> float:
    try:
        return parse_speed(raw_speed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def parse_json_stream_flag(raw_stream: Any) -> bool:
    if raw_stream is None:
        return False
    if isinstance(raw_stream, bool):
        return raw_stream
    raise HTTPException(status_code=400, detail="'stream' must be a boolean.")


def parse_form_stream_flag(raw_stream: Any) -> bool:
    if raw_stream is None or raw_stream == "":
        return False
    value = str(raw_stream).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise HTTPException(status_code=400, detail="'stream' must be 'true' or 'false'.")


async def read_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON payload must be an object.")
    return payload


async def read_form_or_400(request: Request) -> Any:
    try:
        return await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid multipart form data.") from exc


def extract_upload_fields(
    form: Any, *, require_reference_text: bool
) -> tuple[str, str | None, UploadFile, bool, float]:
    text = form.get("text")
    reference_text = form.get("reference_text") if require_reference_text else None
    reference_audio = form.get("reference_audio")
    stream = parse_form_stream_flag(form.get("stream"))
    speed = parse_speed_or_400(form.get("speed", 1.0))

    required: list[tuple[str, Any]] = [("text", text)]
    if require_reference_text:
        required.append(("reference_text", reference_text))
    required.append(("reference_audio", reference_audio))

    missing = [name for name, value in required if not value]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Missing required form fields: {', '.join(missing)}"
        )
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must be a non-empty string.")
    if require_reference_text and (
        not isinstance(reference_text, str) or not reference_text.strip()
    ):
        raise HTTPException(
            status_code=400, detail="Field 'reference_text' must be a non-empty string."
        )
    if not isinstance(reference_audio, UploadFile):
        raise HTTPException(
            status_code=400, detail="Field 'reference_audio' must be a file upload."
        )

    cleaned_ref_text = reference_text.strip() if isinstance(reference_text, str) else None
    return text.strip(), cleaned_ref_text, reference_audio, stream, speed


async def stream_upload_to_file(upload: UploadFile, target: Path, max_bytes: int) -> None:
    total = 0
    file_obj = await asyncio.to_thread(target.open, "wb")
    try:
        while True:
            chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Reference audio exceeds maximum size of {max_bytes} bytes.",
                )
            await asyncio.to_thread(file_obj.write, chunk)
    finally:
        await asyncio.to_thread(file_obj.close)

    if total == 0:
        raise HTTPException(status_code=400, detail="Reference audio file is empty.")


def get_ray_context_ids() -> tuple[str, str]:
    if not ray.is_initialized():
        return "N/A", "N/A"
    try:
        ctx = ray.get_runtime_context()
        actor_id = ctx.get_actor_id() or "N/A"
        node_id = ctx.get_node_id() or "N/A"
        return str(actor_id), str(node_id)
    except Exception:
        return "N/A", "N/A"
