import json
from pathlib import Path

import pytest

from app.core.model_loader import EXPECTED_FEATURE_COLUMNS, ModelStore
from app.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("model")
    result = train(
        TrainConfig(
            artifacts_dir=root / "artifacts",
            production_pool_path=root / "production_pool.csv",
            reference_path=root / "reference.parquet",
        )
    )
    return result["artifact_path"]


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


def test_model_loads_and_predicts(model_dir: Path) -> None:
    store = ModelStore(model_dir)

    predicted_class, probability = store.predict(_features())

    assert store.active_model is not None
    assert store.model_version == model_dir.name
    assert predicted_class in {0, 1}
    assert 0.0 <= probability <= 1.0


def test_wrong_schema_is_refused(model_dir: Path, tmp_path: Path) -> None:
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "model.joblib").write_bytes((model_dir / "model.joblib").read_bytes())
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    schema["columns"] = EXPECTED_FEATURE_COLUMNS[:-1]
    (invalid_dir / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")

    with pytest.raises(ValueError, match="feature schema"):
        ModelStore(invalid_dir)
