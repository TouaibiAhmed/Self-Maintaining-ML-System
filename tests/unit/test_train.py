import subprocess
import sys
from pathlib import Path

import pytest
from pandera.errors import SchemaError, SchemaErrors

from app.training.data_prep import clean, load_raw
from app.training.data_validation import validate_cleaned
from app.training.train import TrainConfig, train

RAW_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def _config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        raw_path=RAW_PATH,
        artifacts_dir=tmp_path / "artifacts",
        production_pool_path=tmp_path / "production_pool.csv",
        reference_path=tmp_path / "reference.parquet",
    )


def test_training_twice_with_same_seed_has_identical_metrics(tmp_path: Path) -> None:
    first = train(_config(tmp_path / "first"))
    second = train(_config(tmp_path / "second"))

    assert first["metrics"] == second["metrics"]


def test_artifact_loads_in_separate_process(tmp_path: Path) -> None:
    result = train(_config(tmp_path))
    artifact_path = result["artifact_path"]
    script = """
import json
import joblib
import pandas as pd
import sys

model = joblib.load(sys.argv[1])
pool = pd.read_csv(sys.argv[2])
features = json.load(open(sys.argv[3], encoding="utf-8"))["columns"]
probabilities = model.predict_proba(pool[features])[:, 1]
assert ((probabilities >= 0) & (probabilities <= 1)).all()
print(len(probabilities))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(artifact_path / "model.joblib"),
            str(tmp_path / "production_pool.csv"),
            str(artifact_path / "feature_schema.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert int(completed.stdout.strip()) > 0


@pytest.mark.parametrize(
    ("column", "value"),
    [("tenure", "not-an-integer"), ("MonthlyCharges", "not-a-number")],
)
def test_invalid_schema_is_rejected(column: str, value: str) -> None:
    cleaned = clean(load_raw(RAW_PATH))
    invalid = cleaned.copy()
    invalid[column] = value

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_cleaned(invalid, reference=cleaned)


def test_missing_schema_column_is_rejected() -> None:
    cleaned = clean(load_raw(RAW_PATH)).drop(columns=["Churn"])

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_cleaned(cleaned)
