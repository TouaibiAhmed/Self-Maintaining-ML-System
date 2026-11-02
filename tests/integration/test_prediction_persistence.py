import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.api import routes
from app.api.main import create_app
from app.core.config import Settings
from app.db.session import engine
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


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    root = tmp_path_factory.mktemp("persistence-model")
    result = train(
        TrainConfig(
            artifacts_dir=root / "artifacts",
            production_pool_path=root / "pool.csv",
            reference_path=root / "reference.parquet",
        )
    )
    return Settings(
        model_dir=result["artifact_path"],
        log_level="INFO",
        max_body_bytes=1024 * 1024,
        database_url=str(engine.url),
    )


@pytest.fixture
def transactional_client(settings: Settings):
    connection = engine.connect()
    transaction = connection.begin()
    session_factory = sessionmaker(bind=connection, autoflush=False, expire_on_commit=False)
    application = create_app(settings)
    with TestClient(application) as client:
        application.state.session_factory = session_factory
        yield client, connection
    transaction.rollback()
    connection.close()


def test_prediction_creates_traceable_event(transactional_client) -> None:
    client, connection = transactional_client

    response = client.post("/predict", json=_features(), headers={"X-Request-ID": "trace-123"})

    assert response.status_code == 200
    prediction_id = response.json()["prediction_id"]
    row = connection.execute(
        text(
            "SELECT prediction_id, feature_payload, predicted_class, "
            "predicted_probability, model_version, request_id, latency_ms "
            "FROM prediction_events WHERE prediction_id = CAST(:prediction_id AS UUID)"
        ),
        {"prediction_id": prediction_id},
    ).mappings().all()
    assert len(row) == 1
    assert str(row[0]["prediction_id"]) == prediction_id
    assert row[0]["feature_payload"]["gender"] == "Female"
    assert row[0]["request_id"] == "trace-123"
    assert row[0]["latency_ms"] >= 0


class FailingSession:
    def add(self, event) -> None:
        self.event = event

    def commit(self) -> None:
        raise SQLAlchemyError("database insert failed")

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class FailingEngine:
    def connect(self):
        raise SQLAlchemyError("database unavailable")


def test_database_insert_failure_keeps_prediction_and_rolls_back(
    settings: Settings,
) -> None:
    application = create_app(settings)
    failing_session = FailingSession()
    with TestClient(application) as client:
        application.state.session_factory = lambda: failing_session
        routes.prediction_log_error_count = 0
        response = client.post("/predict", json=_features())

    assert response.status_code == 200
    assert routes.prediction_log_error_count == 1
    assert failing_session.rolled_back is True
    assert failing_session.closed is True


def test_ready_reflects_database_availability(settings: Settings) -> None:
    application = create_app(settings)
    with TestClient(application) as client:
        application.state.db_engine = FailingEngine()
        response = client.get("/ready")

    assert response.status_code == 503
