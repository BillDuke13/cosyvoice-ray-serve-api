"""Ray Serve deployment declaration and FastAPI ingress binding."""

from __future__ import annotations

from typing import Any

import torch
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from cosyvoice_api.config import DeploymentSettings, Settings
from cosyvoice_api.service import CosyVoiceHandlers, configure_cors


def _build_ray_actor_options(deploy: DeploymentSettings) -> dict[str, Any]:
    if torch.cuda.is_available() and deploy.num_gpus_per_replica > 0:
        return {"num_gpus": deploy.num_gpus_per_replica}
    return {}


http_app = FastAPI(title="CosyVoice Ray Serve API")
configure_cors(http_app)

_startup_settings = Settings.from_env()
_deploy_settings = DeploymentSettings.from_env()


@serve.deployment(
    name="CosyVoiceAPI",
    autoscaling_config={
        "min_replicas": _deploy_settings.min_replicas,
        "initial_replicas": _deploy_settings.initial_replicas,
        "max_replicas": _deploy_settings.max_replicas,
        "target_ongoing_requests": _deploy_settings.target_ongoing_requests,
        "look_back_period_s": 30.0,
        "upscaling_factor": 1.0,
        "downscaling_factor": 1.0,
    },
    ray_actor_options=_build_ray_actor_options(_deploy_settings),
    health_check_period_s=_deploy_settings.health_check_period_s,
    health_check_timeout_s=_deploy_settings.health_check_timeout_s,
    max_ongoing_requests=_deploy_settings.max_ongoing_requests_per_replica,
    max_queued_requests=_deploy_settings.max_queued_requests_deployment,
    graceful_shutdown_timeout_s=_deploy_settings.graceful_shutdown_timeout_s,
    graceful_shutdown_wait_loop_s=_deploy_settings.graceful_shutdown_wait_loop_s,
    user_config=_startup_settings.to_user_config(),
)
@serve.ingress(http_app)
class CosyVoiceService(CosyVoiceHandlers):
    """Ray Serve ingress deployment exposing CosyVoiceHandlers over FastAPI."""

    @http_app.get("/health")
    @http_app.get("/v1/model/cosyvoice/healthcheck")
    async def health_check(self, request: Request) -> JSONResponse:
        return await super().health_check(request)

    @http_app.get("/config")
    async def get_config(self, request: Request) -> JSONResponse:
        return await super().get_config(request)

    @http_app.post("/tts")
    async def tts_standard(self, request: Request) -> Response:
        return await super().tts_standard(request)

    @http_app.post("/zero_shot_tts")
    async def tts_zero_shot(self, request: Request) -> Response:
        return await super().tts_zero_shot(request)

    @http_app.post("/cross_lingual_tts")
    async def tts_cross_lingual(self, request: Request) -> Response:
        return await super().tts_cross_lingual(request)

    @http_app.post("/instruct_tts")
    async def tts_instruct(self, request: Request) -> Response:
        return await super().tts_instruct(request)


app = CosyVoiceService.bind()
