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


def _copy_artifact(model_dir: Path, target: Path, metrics: dict | None) -> Path:
    target.mkdir()
    for name in ("model.joblib", "feature_schema.json"):
        (target / name).write_bytes((model_dir / name).read_bytes())
    if metrics is not None:
        (target / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return target


def test_threshold_comes_from_artifact_metrics(model_dir: Path) -> None:
    store = ModelStore(model_dir)
    expected = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["threshold"]

    assert store.threshold == expected


@pytest.mark.parametrize("threshold", [0.01, 0.99])
def test_predicted_class_uses_artifact_threshold(
    model_dir: Path, tmp_path: Path, threshold: float
) -> None:
    store = ModelStore(_copy_artifact(model_dir, tmp_path / "copy", {"threshold": threshold}))

    predicted_class, probability = store.predict(_features())

    assert predicted_class == int(probability >= threshold)
    assert predicted_class == (1 if threshold == 0.01 else 0)


def test_missing_metrics_is_refused(model_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metrics"):
        ModelStore(_copy_artifact(model_dir, tmp_path / "copy", None))


@pytest.mark.parametrize(
    "metrics", [{}, {"threshold": "0.5"}, {"threshold": 0}, {"threshold": 1.5}]
)
def test_invalid_threshold_is_refused(model_dir: Path, tmp_path: Path, metrics: dict) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ModelStore(_copy_artifact(model_dir, tmp_path / "copy", metrics))
