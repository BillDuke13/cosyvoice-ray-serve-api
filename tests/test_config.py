from __future__ import annotations

from pathlib import Path

from cosyvoice_api.config import (
    DEFAULT_MODEL_DIR_NAME,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_VOICE,
    REFERENCE_AUDIO_SAMPLE_RATE,
    ROOT_DIR,
    SAMPLE_RATE,
    DeploymentSettings,
    Settings,
)


def test_settings_defaults_and_no_side_effects(tmp_path: Path) -> None:
    settings = Settings.from_env({}, root_dir=tmp_path)

    assert settings.sample_rate == SAMPLE_RATE == 24000
    assert settings.reference_audio_sample_rate == REFERENCE_AUDIO_SAMPLE_RATE == 16000
    assert settings.default_voice == DEFAULT_VOICE == "qwen"
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.model_dir_name == DEFAULT_MODEL_DIR_NAME
    assert settings.model_revision == DEFAULT_MODEL_REVISION
    assert settings.serve_port == 8000
    assert settings.voices_dir == tmp_path.resolve() / "assets" / "voices"
    assert settings.voice_prompts["qwen"] == tmp_path.resolve() / "assets" / "voices" / "qwen.wav"

    # Constructing settings must not touch the filesystem until explicitly requested.
    assert not settings.tmp_dir.exists()
    assert not settings.logs_dir.exists()

    settings.ensure_runtime_dirs()
    assert settings.tmp_dir.is_dir()
    assert settings.logs_dir.is_dir()
    assert settings.voices_dir.is_dir()
    assert settings.model_dir.parent.is_dir()


def test_settings_env_overrides(tmp_path: Path) -> None:
    env = {
        "COSYVOICE_MODEL_ID": "Org/Custom-Model",
        "COSYVOICE_MODEL_DIR_NAME": "Custom-Model",
        "COSYVOICE_MODEL_REVISION": "v1.0",
        "SERVE_PORT": "9998",
        "MAX_REPLICAS": "4",
    }
    settings = Settings.from_env(env, root_dir=tmp_path)

    assert settings.model_id == "Org/Custom-Model"
    assert settings.model_dir == tmp_path.resolve() / "pretrained_models" / "Custom-Model"
    assert settings.model_revision == "v1.0"
    assert settings.serve_port == 9998
    assert settings.max_replicas == 4


def test_deployment_settings_env_overrides() -> None:
    deploy = DeploymentSettings.from_env(
        {
            "MIN_REPLICAS": "2",
            "INITIAL_REPLICAS": "2",
            "MAX_REPLICAS": "6",
            "TARGET_ONGOING_REQUESTS": "3",
            "NUM_GPUS_PER_REPLICA": "0",
            "HEALTH_CHECK_PERIOD_S": "10",
            "HEALTH_CHECK_TIMEOUT_S": "20",
            "MAX_ONGOING_REQUESTS_PER_REPLICA": "8",
            "MAX_QUEUED_REQUESTS_DEPLOYMENT": "40",
            "GRACEFUL_SHUTDOWN_TIMEOUT_S": "90",
            "GRACEFUL_SHUTDOWN_WAIT_LOOP_S": "2",
        }
    )

    assert deploy.min_replicas == 2
    assert deploy.initial_replicas == 2
    assert deploy.max_replicas == 6
    assert deploy.target_ongoing_requests == 3
    assert deploy.num_gpus_per_replica == 0
    assert deploy.health_check_period_s == 10
    assert deploy.health_check_timeout_s == 20
    assert deploy.max_ongoing_requests_per_replica == 8
    assert deploy.max_queued_requests_deployment == 40
    assert deploy.graceful_shutdown_timeout_s == 90
    assert deploy.graceful_shutdown_wait_loop_s == 2


def test_default_voice_asset_present_in_repo() -> None:
    assert (ROOT_DIR / "assets" / "voices" / "qwen.wav").is_file()
