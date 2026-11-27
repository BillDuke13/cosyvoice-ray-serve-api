"""Runtime settings, deployment defaults, and directory layout."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fixed sample rates required by CosyVoice3: 24 kHz output, 16 kHz mono reference clips.
SAMPLE_RATE = 24000
REFERENCE_AUDIO_SAMPLE_RATE = 16000

MODEL_FAMILY = "cosyvoice3"
DEFAULT_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_MODEL_DIR_NAME = "Fun-CosyVoice3-0.5B"
# ModelScope uses "master" as the default branch for this repository.
DEFAULT_MODEL_REVISION = "master"

END_OF_PROMPT = "<|endofprompt|>"
SYSTEM_PROMPT = "You are a helpful assistant."

MIN_TTS_SPEED = 0.5
MAX_TTS_SPEED = 2.0

# Maps voice_type keys to WAV filenames under assets/voices/.
VOICE_PROMPTS: dict[str, str] = {"qwen": "qwen.wav"}
DEFAULT_VOICE = "qwen"

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    return int(env.get(key, str(default)))


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
    def from_env(cls, env: Mapping[str, str] | None = None) -> DeploymentSettings:
        source = os.environ if env is None else env
        return cls(
            min_replicas=_env_int(source, "MIN_REPLICAS", 1),
            initial_replicas=_env_int(source, "INITIAL_REPLICAS", 1),
            max_replicas=_env_int(source, "MAX_REPLICAS", 2),
            target_ongoing_requests=_env_int(source, "TARGET_ONGOING_REQUESTS", 2),
            num_gpus_per_replica=_env_int(source, "NUM_GPUS_PER_REPLICA", 1),
            health_check_period_s=_env_int(source, "HEALTH_CHECK_PERIOD_S", 15),
            health_check_timeout_s=_env_int(source, "HEALTH_CHECK_TIMEOUT_S", 30),
            max_ongoing_requests_per_replica=_env_int(
                source, "MAX_ONGOING_REQUESTS_PER_REPLICA", 5
            ),
            max_queued_requests_deployment=_env_int(source, "MAX_QUEUED_REQUESTS_DEPLOYMENT", 20),
            graceful_shutdown_timeout_s=_env_int(source, "GRACEFUL_SHUTDOWN_TIMEOUT_S", 60),
            graceful_shutdown_wait_loop_s=_env_int(source, "GRACEFUL_SHUTDOWN_WAIT_LOOP_S", 5),
        )


@dataclass(frozen=True)
class Settings:
    """Application paths, model coordinates, and audio configuration."""

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
    voice_prompts: dict[str, Path]

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        root_dir: Path | None = None,
    ) -> Settings:
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
            serve_port=_env_int(source, "SERVE_PORT", 8000),
            max_replicas=_env_int(source, "MAX_REPLICAS", 2),
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
