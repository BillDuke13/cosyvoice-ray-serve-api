"""CosyVoice3 Ray Serve HTTP API package."""

from typing import Any

from cosyvoice_api.config import DeploymentSettings, Settings, configure_logging

__all__ = [
    "CosyVoiceHandlers",
    "CosyVoiceService",
    "DeploymentSettings",
    "Settings",
    "app",
    "configure_logging",
    "create_asgi_app",
    "http_app",
]


def __getattr__(name: str) -> Any:
    if name in {"CosyVoiceService", "app", "http_app"}:
        from cosyvoice_api import deployment

        return getattr(deployment, name)
    if name in {"CosyVoiceHandlers", "create_asgi_app"}:
        from cosyvoice_api import service

        return getattr(service, name)
    raise AttributeError(f"module 'cosyvoice_api' has no attribute {name!r}")
