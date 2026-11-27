from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from cosyvoice_api.config import Settings
from cosyvoice_api.engine import (
    REQUIRED_MODEL_FILES,
    CosyVoiceEngine,
    configure_cuda_environment,
    model_files_present,
    resolve_local_cuda_ordinal,
)


def _populate_model_dir(model_dir: Path) -> None:
    for rel_path in REQUIRED_MODEL_FILES:
        file_path = model_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"stub")


def test_model_files_present_requires_all_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    assert model_files_present(model_dir) is False
    _populate_model_dir(model_dir)
    assert model_files_present(model_dir) is True

    (model_dir / "CosyVoice-BlankEN" / "config.json").unlink()
    assert model_files_present(model_dir) is False


def test_resolve_local_cuda_ordinal_handles_isolation_and_uuids() -> None:
    # Single visible GPU in Ray-isolated actor always maps to local ordinal 0.
    assert resolve_local_cuda_ordinal(1, device_count=1) == 0
    assert resolve_local_cuda_ordinal("GPU-abc-123", device_count=1) == 0

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,GPU-bbb"}):
        assert resolve_local_cuda_ordinal("GPU-bbb", device_count=2) == 1
        assert resolve_local_cuda_ordinal("GPU-aaa", device_count=2) == 0

    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        assert resolve_local_cuda_ordinal(1, device_count=2) == 1
        assert resolve_local_cuda_ordinal(5, device_count=2) == 0
        assert resolve_local_cuda_ordinal("non-numeric", device_count=2) == 0


def test_configure_cuda_environment_respects_existing_overrides() -> None:
    with patch.dict(os.environ, {"CUDA_LAUNCH_BLOCKING": "1"}, clear=True):
        configure_cuda_environment()
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_LAUNCH_BLOCKING"] == "1"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"


def test_engine_select_device_cpu_when_cuda_unavailable(tmp_path: Path) -> None:
    settings = Settings.from_env({}, root_dir=tmp_path)
    engine = CosyVoiceEngine(settings)
    with patch("cosyvoice_api.engine.torch.cuda.is_available", return_value=False):
        assert engine.select_device() == "cpu"


def test_engine_load_downloads_missing_model_and_retries_fp32(tmp_path: Path) -> None:
    settings = Settings.from_env({}, root_dir=tmp_path)
    engine = CosyVoiceEngine(settings)
    fake_model = MagicMock()

    def fake_download(_model_id: str, local_dir: str, revision: str) -> None:
        assert revision == settings.model_revision
        _populate_model_dir(Path(local_dir))

    auto_model_calls: list[dict[str, object]] = []

    def fake_auto_model(**kwargs: object) -> object:
        auto_model_calls.append(kwargs)
        if kwargs.get("fp16") is True:
            raise RuntimeError("fp16 init failure")
        return fake_model

    with (
        patch("cosyvoice_api.engine.configure_torch_cuda"),
        patch.object(engine, "select_device", return_value="cuda:0"),
        patch("modelscope.snapshot_download", side_effect=fake_download) as download_mock,
        patch("cosyvoice.cli.cosyvoice.AutoModel", side_effect=fake_auto_model),
    ):
        engine.load()

    assert download_mock.call_count == 1
    assert [c["fp16"] for c in auto_model_calls] == [True, False]
    assert engine.is_loaded is True
    assert engine.device == "cuda:0"


def test_engine_iter_synthesize_and_close(tmp_path: Path) -> None:
    settings = Settings.from_env({}, root_dir=tmp_path)
    engine = CosyVoiceEngine(settings)

    with pytest.raises(RuntimeError, match="not loaded"):
        list(engine.iter_speech_chunks("inference_cross_lingual", "hi", stream=False))

    def fake_inference(_text: str, **_kwargs: object) -> list[dict[str, object]]:
        return [
            {"tts_speech": torch.tensor([[0.0, float("nan")]])},  # invalid -> skipped
            {"tts_speech": torch.tensor([[0.25, -0.5]])},
            {"tts_speech": torch.tensor([[0.1, 0.2]])},
        ]

    engine.model = SimpleNamespace(inference_cross_lingual=fake_inference)
    combined = engine.synthesize("inference_cross_lingual", "hello")
    assert combined.shape == (1, 4)
    assert torch.isclose(combined[:, :2].abs().max(), torch.tensor(1.0))

    with pytest.raises(AttributeError):
        engine.synthesize("nonexistent_method", "hello")

    engine.model = SimpleNamespace(empty_method=lambda _t, **_kw: [{"tts_speech": torch.empty(0)}])
    with pytest.raises(RuntimeError, match="no valid audio|failed to produce valid audio"):
        engine.synthesize("empty_method", "hello")

    engine.close()
    assert engine.is_loaded is False
