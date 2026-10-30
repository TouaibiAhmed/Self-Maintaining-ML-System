import json
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.config import Settings
from app.training.train import TrainConfig, train


def _features() -> dict[str, object]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def _settings(tmp_path: Path) -> Settings:
    result = train(
        TrainConfig(
            artifacts_dir=tmp_path / "artifacts",
            production_pool_path=tmp_path / "pool.csv",
            reference_path=tmp_path / "reference.parquet",
        )
    )
    return Settings(
        model_dir=result["artifact_path"],
        log_level="INFO",
        max_body_bytes=1024,
    )


def test_valid_prediction_returns_uuid(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/predict", json=_features())

    assert response.status_code == 200
    body = response.json()
    UUID(body["prediction_id"])
    assert body["predicted_class"] in {0, 1}
    assert 0 <= body["probability"] <= 1


def test_missing_field_returns_422(tmp_path: Path) -> None:
    payload = _features()
    del payload["gender"]
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_wrong_type_returns_422(tmp_path: Path) -> None:
    payload = _features()
    payload["tenure"] = "not-an-integer"
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_oversized_body_is_rejected(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/predict", content=json.dumps(_features()) + " " * 1024)

    assert response.status_code == 413


def test_model_load_failure_returns_503(tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, log_level="INFO", max_body_bytes=1024)
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        prediction = client.post("/predict", json=_features())

    assert ready.status_code == 503
    assert prediction.status_code == 503
