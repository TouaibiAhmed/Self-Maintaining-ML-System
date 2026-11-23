"""Safe loading and inference for the active serialized model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

EXPECTED_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

EXPECTED_FEATURE_DTYPES = {
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
}


class ModelStore:
    """Load one validated model and keep it available for inference."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.schema_hash = self._schema_hash()
        self.active_model: Any = self._load()
        self.threshold = self._load_threshold()
        self.model_version = self.model_dir.name
        self.loaded_at = datetime.now(timezone.utc)

    def _load_threshold(self) -> float:
        """Read the decision threshold chosen during training, so serving matches evaluation."""
        metrics_path = self.model_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing model metrics: {metrics_path}")
        threshold = json.loads(metrics_path.read_text(encoding="utf-8")).get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, int | float):
            raise ValueError(f"metrics.json has no numeric decision threshold: {metrics_path}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Decision threshold must be between 0 and 1, got {threshold}")
        return float(threshold)

    def _load(self) -> Any:
        schema_path = self.model_dir / "feature_schema.json"
        model_path = self.model_dir / "model.joblib"
        if not schema_path.is_file():
            raise FileNotFoundError(f"Missing model schema: {schema_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing serialized model: {model_path}")

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        columns = schema.get("columns")
        dtypes = schema.get("dtypes")
        if columns != EXPECTED_FEATURE_COLUMNS or dtypes != EXPECTED_FEATURE_DTYPES:
            raise ValueError(
                "Model feature schema does not match the API feature contract"
            )
        return joblib.load(model_path)

    def _schema_hash(self) -> str:
        schema_path = self.model_dir / "feature_schema.json"
        if not schema_path.is_file():
            raise FileNotFoundError(f"Missing model schema: {schema_path}")
        return hashlib.sha256(schema_path.read_bytes()).hexdigest()

    def predict(self, features_dict: dict[str, Any]) -> tuple[int, float]:
        """Predict one row after validating its exact feature names."""
        received = list(features_dict)
        if set(received) != set(EXPECTED_FEATURE_COLUMNS):
            missing = sorted(set(EXPECTED_FEATURE_COLUMNS) - set(received))
            extra = sorted(set(received) - set(EXPECTED_FEATURE_COLUMNS))
            raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")

        features = pd.DataFrame(
            [[features_dict[column] for column in EXPECTED_FEATURE_COLUMNS]],
            columns=EXPECTED_FEATURE_COLUMNS,
        )
        probability = float(self.active_model.predict_proba(features)[0, 1])
        predicted_class = int(probability >= self.threshold)
        return predicted_class, probability
