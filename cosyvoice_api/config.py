"""Runtime settings, deployment defaults, logging setup, and directory layout."""

import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

# Fixed sample rates required by CosyVoice3: 24 kHz output, 16 kHz mono reference clips.
SAMPLE_RATE = 24000
REFERENCE_AUDIO_SAMPLE_RATE = 16000
MAX_REFERENCE_AUDIO_DURATION_S = 30

MODEL_FAMILY = "cosyvoice3"
DEFAULT_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_MODEL_DIR_NAME = "Fun-CosyVoice3-0.5B"
# ModelScope uses "master" as the default branch for this repository.
DEFAULT_MODEL_REVISION = "master"

END_OF_PROMPT = "<|endofprompt|>"
SYSTEM_PROMPT = "You are a helpful assistant."

MIN_TTS_SPEED = 0.5
MAX_TTS_SPEED = 2.0
DEFAULT_MAX_REFERENCE_AUDIO_BYTES = 25 * 1024 * 1024
DEFAULT_CORS_ALLOW_ORIGINS = ("*",)

# Maps voice_type keys to WAV filenames under assets/voices/.
VOICE_PROMPTS: dict[str, str] = {"qwen": "qwen.wav"}
DEFAULT_VOICE = "qwen"

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_int(env: Mapping[str, str], key: str, default: int, *, minimum: int | None = None) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {key} must be an integer, got {raw!r}."
            ) from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {key} must be >= {minimum}, got {value}.")
    return value


def _env_csv(
    env: Mapping[str, str],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or default


@dataclass(frozen=True)
class DeploymentSettings:
    """Ray Serve autoscaling, resource, health-check, and shutdown settings."""

    min_replicas: int = 1
    initial_replicas: int = 1
    max_replicas: int = 2
    target_ongoing_requests: int = 2
    num_gpus_per_replica: int = 1
    health_check_period_s: int = 15
    health_check_timeout_s: int = 30
    max_ongoing_requests_per_replica: int = 5
    max_queued_requests_deployment: int = 20
    graceful_shutdown_timeout_s: int = 60
    graceful_shutdown_wait_loop_s: int = 5

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        source = os.environ if env is None else env
        min_replicas = _env_int(source, "MIN_REPLICAS", 1, minimum=0)
        initial_replicas = _env_int(source, "INITIAL_REPLICAS", 1, minimum=0)
        max_replicas = _env_int(source, "MAX_REPLICAS", 2, minimum=1)
        if min_replicas > max_replicas:
            raise ValueError(
                f"MIN_REPLICAS ({min_replicas}) cannot exceed MAX_REPLICAS ({max_replicas})."
            )
        return cls(
            min_replicas=min_replicas,
            initial_replicas=initial_replicas,
            max_replicas=max_replicas,
            target_ongoing_requests=_env_int(source, "TARGET_ONGOING_REQUESTS", 2, minimum=1),
            num_gpus_per_replica=_env_int(source, "NUM_GPUS_PER_REPLICA", 1, minimum=0),
            health_check_period_s=_env_int(source, "HEALTH_CHECK_PERIOD_S", 15, minimum=1),
            health_check_timeout_s=_env_int(source, "HEALTH_CHECK_TIMEOUT_S", 30, minimum=1),
            max_ongoing_requests_per_replica=_env_int(
                source, "MAX_ONGOING_REQUESTS_PER_REPLICA", 5, minimum=1
            ),
            max_queued_requests_deployment=_env_int(
                source, "MAX_QUEUED_REQUESTS_DEPLOYMENT", 20, minimum=-1
            ),
            graceful_shutdown_timeout_s=_env_int(
                source, "GRACEFUL_SHUTDOWN_TIMEOUT_S", 60, minimum=1
            ),
            graceful_shutdown_wait_loop_s=_env_int(
                source, "GRACEFUL_SHUTDOWN_WAIT_LOOP_S", 5, minimum=1
            ),
        )


@dataclass(frozen=True)
class Settings:
    """Application paths, model coordinates, limits, and audio configuration."""

    root_dir: Path
    tmp_dir: Path
    logs_dir: Path
    voices_dir: Path
    model_dir: Path
    model_id: str
    model_dir_name: str
    model_revision: str
    model_family: str
    default_voice: str
    sample_rate: int
    reference_audio_sample_rate: int
    serve_port: int
    max_replicas: int
    max_reference_audio_bytes: int
    cors_allow_origins: tuple[str, ...]
    voice_prompts: dict[str, Path]

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        root_dir: Path | None = None,
    ) -> Self:
        source = os.environ if env is None else env
        base = ROOT_DIR if root_dir is None else root_dir.resolve()
        model_id = source.get("COSYVOICE_MODEL_ID", DEFAULT_MODEL_ID)
        model_dir_name = source.get("COSYVOICE_MODEL_DIR_NAME", DEFAULT_MODEL_DIR_NAME)
        model_revision = source.get("COSYVOICE_MODEL_REVISION", DEFAULT_MODEL_REVISION)
        voices_dir = base / "assets" / "voices"
        voice_prompts = {
            key: voices_dir / Path(filename).name for key, filename in VOICE_PROMPTS.items()
        }

        return cls(
            root_dir=base,
            tmp_dir=base / "tmp",
            logs_dir=base / "logs",
            voices_dir=voices_dir,
            model_dir=base / "pretrained_models" / model_dir_name,
            model_id=model_id,
            model_dir_name=model_dir_name,
            model_revision=model_revision,
            model_family=MODEL_FAMILY,
            default_voice=DEFAULT_VOICE,
            sample_rate=SAMPLE_RATE,
            reference_audio_sample_rate=REFERENCE_AUDIO_SAMPLE_RATE,
            serve_port=_env_int(source, "SERVE_PORT", 8000, minimum=1),
            max_replicas=_env_int(source, "MAX_REPLICAS", 2, minimum=1),
            max_reference_audio_bytes=_env_int(
                source,
                "MAX_REFERENCE_AUDIO_BYTES",
                DEFAULT_MAX_REFERENCE_AUDIO_BYTES,
                minimum=1,
            ),
            cors_allow_origins=_env_csv(
                source,
                "CORS_ALLOW_ORIGINS",
                DEFAULT_CORS_ALLOW_ORIGINS,
            ),
            voice_prompts=voice_prompts,
        )

    def ensure_runtime_dirs(self) -> None:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)

    def to_user_config(self) -> dict[str, Any]:
        return {
            "default_voice": self.default_voice,
            "sample_rate": self.sample_rate,
            "model_id": self.model_id,
            "model_dir_name": self.model_dir_name,
            "model_revision": self.model_revision,
            "model_family": self.model_family,
        }


def configure_logging(settings: Settings) -> Path:
    """Attach stdout and file handlers idempotently for driver and Ray replica processes."""
    settings.ensure_runtime_dirs()
    log_path = (settings.logs_dir / "cosyvoice_api.log").resolve()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    has_stdout = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        and getattr(handler, "stream", None) is sys.stdout
        for handler in root.handlers
    )
    if not has_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root.addHandler(stdout_handler)

    has_file = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in root.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return log_path
