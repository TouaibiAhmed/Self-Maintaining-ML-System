"""HTTP routes for model inference and service status."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import CustomerFeatures, PredictionResponse

router = APIRouter()


def log_prediction(event: dict[str, Any]) -> None:
    """Hook for Part 3 prediction persistence; intentionally does nothing yet."""


@router.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures, request: Request) -> PredictionResponse:
    """Return a churn prediction from the active model."""
    model_store = getattr(request.app.state, "model_store", None)
    if model_store is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    predicted_class, probability = model_store.predict(features.model_dump())
    prediction_id = uuid4()
    log_prediction(
        {
            "prediction_id": prediction_id,
            "request_id": request.state.request_id,
            "model_version": model_store.model_version,
        }
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
    """Return readiness only when the model loaded successfully."""
    model_store = getattr(request.app.state, "model_store", None)
    if model_store is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
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
