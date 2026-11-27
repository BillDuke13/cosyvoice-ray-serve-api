"""Local CLI entrypoint (`python -m cosyvoice_api`)."""

from __future__ import annotations

import logging
import sys

import ray
from ray import serve

from cosyvoice_api.config import Settings
from cosyvoice_api.deployment import app
from cosyvoice_api.engine import configure_cuda_environment


def configure_logging(settings: Settings) -> None:
    settings.ensure_runtime_dirs()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(settings.logs_dir / "cosyvoice_api.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)


def main() -> None:
    configure_cuda_environment()
    settings = Settings.from_env()
    configure_logging(settings)

    logger = logging.getLogger(__name__)
    logger.info("Starting local Ray Serve instance on port %d", settings.serve_port)

    if not ray.is_initialized():
        ray.init()

    serve.start(http_options={"host": "0.0.0.0", "port": settings.serve_port})
    serve.run(app, name="cosyvoice_serve_app", blocking=True)


if __name__ == "__main__":
    main()
