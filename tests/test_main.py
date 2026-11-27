from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cosyvoice_api.__main__ import _configure_cuda_env_before_torch, _install_signal_handlers, main
from cosyvoice_api.config import Settings


def test_configure_cuda_env_before_torch_sets_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        _configure_cuda_env_before_torch()
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_LAUNCH_BLOCKING"] == "0"
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"


def test_sigterm_handler_raises_keyboard_interrupt() -> None:
    registered: dict[int, object] = {}

    def fake_signal(sig: int, handler: object) -> None:
        registered[sig] = handler

    with patch("cosyvoice_api.__main__.signal.signal", side_effect=fake_signal):
        _install_signal_handlers()

    handler = registered[signal.SIGTERM]
    assert callable(handler)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)


def test_main_runs_serve_and_shuts_down_cleanly(tmp_path: Path) -> None:
    settings = Settings.from_env({"SERVE_PORT": "9123"}, root_dir=tmp_path)
    fake_ray = MagicMock()
    fake_ray.is_initialized.side_effect = [False, True]
    fake_serve = MagicMock()
    fake_serve.run.side_effect = KeyboardInterrupt("stop")
    fake_app = object()

    with (
        patch("cosyvoice_api.__main__.Settings.from_env", return_value=settings),
        patch("cosyvoice_api.__main__._install_signal_handlers"),
        patch.dict(
            "sys.modules",
            {
                "ray": fake_ray,
                "cosyvoice_api.deployment": MagicMock(app=fake_app),
            },
        ),
    ):
        fake_ray.serve = fake_serve
        main()

    fake_ray.init.assert_called_once()
    fake_serve.start.assert_called_once_with(http_options={"host": "0.0.0.0", "port": 9123})
    fake_serve.run.assert_called_once_with(fake_app, name="cosyvoice_serve_app", blocking=True)
    fake_serve.shutdown.assert_called_once()
    fake_ray.shutdown.assert_called_once()
