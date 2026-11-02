"""HTTP routes for model inference and service status."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.schemas import CustomerFeatures, PredictionResponse
from app.db.models import PredictionEvent

router = APIRouter()
logger = logging.getLogger(__name__)
prediction_log_error_count = 0


def log_prediction(event: dict[str, Any], session_factory: Any) -> bool:
    """Persist one prediction event, rolling back and reporting failures."""
    global prediction_log_error_count
    if session_factory is None:
        prediction_log_error_count += 1
        logger.error(
            "prediction_event_insert_failed",
            extra={"request_id": event["request_id"], "event": "prediction_event_insert_failed"},
        )
        return False
    session = session_factory()
    try:
        session.add(PredictionEvent(**event))
        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        prediction_log_error_count += 1
        logger.exception(
            "prediction_event_insert_failed",
            extra={"request_id": event["request_id"], "event": "prediction_event_insert_failed"},
        )
        return False
    finally:
        session.close()


@router.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures, request: Request) -> PredictionResponse:
    """Return a churn prediction from the active model."""
    model_store = getattr(request.app.state, "model_store", None)
    if model_store is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    started = time.perf_counter()
    feature_payload = features.model_dump()
    predicted_class, probability = model_store.predict(feature_payload)
    prediction_id = uuid4()
    log_prediction(
        {
            "prediction_id": prediction_id,
            "event_time": request.state.event_time,
            "feature_payload": feature_payload,
            "predicted_class": predicted_class,
            "predicted_probability": probability,
            "model_version": model_store.model_version,
            "request_id": request.state.request_id,
            "latency_ms": (time.perf_counter() - started) * 1000,
        },
        request.app.state.session_factory,
    )
    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_class=predicted_class,
        probability=probability,
        model_version=model_store.model_version,
    )


@router.get("/health")
def health() -> dict[str, str]:
    """Return process liveness without checking dependencies."""
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request) -> dict[str, str]:
    """Return readiness only when both model and database are available."""
    model_store = getattr(request.app.state, "model_store", None)
    if model_store is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Database is not configured")
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database is unavailable") from None
    return {"status": "ready"}


@router.get("/model-info")
def model_info(request: Request) -> dict[str, str]:
    """Return metadata for the active model."""
    model_store = getattr(request.app.state, "model_store", None)
    if model_store is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "version": model_store.model_version,
        "loaded_at": model_store.loaded_at.isoformat(),
        "schema_hash": model_store.schema_hash,
    }


@router.get("/metrics")
def metrics() -> dict[str, str]:
    """Placeholder metrics endpoint until observability is implemented."""
    # Part 5 will expose Prometheus-compatible metrics here.
    return {"status": "metrics placeholder"}
