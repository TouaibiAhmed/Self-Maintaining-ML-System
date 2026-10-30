"""Deterministic baseline training and artifact packaging."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.training.data_prep import clean, load_raw
from app.training.data_validation import REQUIRED_COLUMNS, validate_cleaned

DEFAULT_RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_ARTIFACTS_PATH = Path("artifacts")
DEFAULT_PRODUCTION_POOL_PATH = Path("data/processed/production_pool.csv")
DEFAULT_REFERENCE_PATH = Path("data/reference/reference.parquet")
DEFAULT_SEED = 42
THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class TrainConfig:
    """Paths and reproducibility settings for a training run."""

    raw_path: Path = DEFAULT_RAW_PATH
    artifacts_dir: Path = DEFAULT_ARTIFACTS_PATH
    production_pool_path: Path = DEFAULT_PRODUCTION_POOL_PATH
    reference_path: Path = DEFAULT_REFERENCE_PATH
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_path", Path(self.raw_path))
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))
        object.__setattr__(self, "production_pool_path", Path(self.production_pool_path))
        object.__setattr__(self, "reference_path", Path(self.reference_path))


def _config_from_input(config: TrainConfig | Mapping[str, Any] | None) -> TrainConfig:
    if config is None:
        return TrainConfig()
    if isinstance(config, TrainConfig):
        return config
    return TrainConfig(**config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_value) + "\n", encoding="utf-8")


def _metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": matrix,
        "positive_prevalence": float(np.mean(y_true)),
        "sample_count": int(len(y_true)),
        "positive_count": int(np.sum(y_true)),
        "negative_count": int(len(y_true) - np.sum(y_true)),
    }


def _make_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def train(config: TrainConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Train the baseline model and return its artifact directory and metrics."""
    settings = _config_from_input(config)
    raw = load_raw(settings.raw_path)
    cleaned = clean(raw)
    validate_cleaned(cleaned)
    cleaned.insert(0, "row_id", np.arange(len(cleaned), dtype=np.int64))

    production_pool, remaining = train_test_split(
        cleaned,
        test_size=0.70,
        stratify=cleaned["Churn"],
        random_state=settings.seed,
    )
    settings.production_pool_path.parent.mkdir(parents=True, exist_ok=True)
    production_pool.to_csv(settings.production_pool_path, index=False)

    train_df, holdout = train_test_split(
        remaining,
        test_size=0.30,
        stratify=remaining["Churn"],
        random_state=settings.seed,
    )
    validation_df, test_df = train_test_split(
        holdout,
        test_size=0.50,
        stratify=holdout["Churn"],
        random_state=settings.seed,
    )

    feature_columns = [
        column for column in REQUIRED_COLUMNS if column not in {"customerID", "Churn"}
    ]
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    pipeline = _make_pipeline(numeric_columns, categorical_columns)
    pipeline.fit(train_df[feature_columns], train_df["Churn"])

    validation_probabilities = pipeline.predict_proba(validation_df[feature_columns])[:, 1]
    threshold = max(
        THRESHOLD_GRID,
        key=lambda candidate: (
            f1_score(
                validation_df["Churn"],
                validation_probabilities >= candidate,
                zero_division=0,
            ),
            -candidate,
        ),
    )
    test_probabilities = pipeline.predict_proba(test_df[feature_columns])[:, 1]
    metrics = _metrics(test_df["Churn"], test_probabilities, threshold)
    metrics["split_sizes"] = {
        "production_pool": len(production_pool),
        "train": len(train_df),
        "validation": len(validation_df),
        "test": len(test_df),
    }

    timestamp = datetime.now(timezone.utc)
    version = timestamp.strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = settings.artifacts_dir / version
    suffix = 1
    while artifact_dir.exists():
        artifact_dir = settings.artifacts_dir / f"{version}-{suffix}"
        suffix += 1
    artifact_dir.mkdir(parents=True)
    joblib.dump(pipeline, artifact_dir / "model.joblib")
    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(
        artifact_dir / "feature_schema.json",
        {
            "columns": feature_columns,
            "dtypes": {column: str(cleaned[column].dtype) for column in feature_columns},
        },
    )
    _write_json(
        artifact_dir / "run_metadata.json",
        {
            "seed": settings.seed,
            "raw_sha256": _sha256(settings.raw_path),
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "library_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "joblib": package_version("joblib"),
            },
            "split_sizes": metrics["split_sizes"],
            "timestamp_utc": timestamp.isoformat(),
            "config": asdict(settings),
        },
    )

    if not settings.reference_path.exists():
        settings.reference_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(settings.reference_path, index=False)

    return {"artifact_path": artifact_dir, "metrics": metrics}
