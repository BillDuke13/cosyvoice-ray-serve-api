"""Local CLI entrypoint (`python -m cosyvoice_api`)."""

import logging
import os
import signal
from types import FrameType

from cosyvoice_api.config import Settings, configure_logging


def _configure_cuda_env_before_torch() -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")


def _install_signal_handlers() -> None:
    def _handle_stop(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_stop)


def main() -> None:
    # Apply CUDA env vars before importing torch or cosyvoice_api.deployment.
    _configure_cuda_env_before_torch()
    settings = Settings.from_env()
    configure_logging(settings)
    _install_signal_handlers()

    import ray
    from ray import serve

    from cosyvoice_api.deployment import app

    logger = logging.getLogger(__name__)
    logger.info("Starting local Ray Serve instance on port %d", settings.serve_port)

    if not ray.is_initialized():
        ray.init()

    try:
        serve.start(http_options={"host": "0.0.0.0", "port": settings.serve_port})
        serve.run(app, name="cosyvoice_serve_app", blocking=True)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received; stopping Ray Serve.")
    finally:
        try:
            serve.shutdown()
        finally:
            if ray.is_initialized():
                ray.shutdown()


if __name__ == "__main__":
    main()
