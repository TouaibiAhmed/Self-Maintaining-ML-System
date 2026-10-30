"""FastAPI application setup and request middleware."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings
from app.core.logging import configure_logging, get_logger, log_event
from app.core.model_loader import ModelStore

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application with optional settings injection for tests."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime_settings = settings or Settings()
        configure_logging(runtime_settings.log_level)
        application.state.model_store = None
        try:
            application.state.model_store = ModelStore(runtime_settings.model_dir)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.error(
                "model_load_failed",
                extra={"request_id": "-", "event": "model_load_failed"},
                exc_info=exc,
            )
        yield

    application = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)

    @application.middleware("http")
    async def request_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        max_body_bytes = (settings or Settings()).max_body_bytes
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body exceeds MAX_BODY_BYTES", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        try:
            body = await request.body()
            if len(body) > max_body_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": "Request body exceeds MAX_BODY_BYTES",
                        "request_id": request_id,
                    },
                    headers={"X-Request-ID": request_id},
                )
            response = await call_next(request)
        except Exception:
            log_event(
                logger,
                40,
                "request_failed",
                request_id=request_id,
                event="request_failed",
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        response.headers["X-Request-ID"] = request_id
        log_event(
            logger,
            20,
            "request_completed",
            request_id=request_id,
            event="request_completed",
            status_code=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return response

    application.include_router(router)
    return application


app = create_app()
