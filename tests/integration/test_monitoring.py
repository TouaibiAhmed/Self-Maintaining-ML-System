"""Monitoring job tests against Postgres.

Every test runs in a transaction that is rolled back. The fixture first deletes existing
prediction events INSIDE that transaction, so each window contains only the test's rows and the
development database is left exactly as it was.

The drift threshold here is an explicit test value (0.10), not the configured default: with only
2 of 19 features shifted, feature_drift has a drift share of about 0.105, so the uncalibrated
default of 0.30 cannot detect it. Part 4.3 calibrates the real threshold from measured runs.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, select

from app.db.models import MonitoringRun, OutcomeLabel, PredictionEvent
from app.db.session import engine
from app.monitoring.drift_report import FEATURE_COLUMNS
from app.monitoring.run_monitor import MonitorConfig, run_loop, run_monitoring
from app.training.train import TrainConfig, train
from simulator.replay import build_traffic, request_body

BASE_TIME = datetime(2100, 1, 1, tzinfo=timezone.utc)
TEST_DRIFT_THRESHOLD = 0.10


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("monitoring")
    result = train(
        TrainConfig(
            artifacts_dir=root / "artifacts",
            production_pool_path=root / "pool.csv",
            reference_path=root / "reference.parquet",
        )
    )
    model = joblib.load(result["artifact_path"] / "model.joblib")
    return {
        "root": root,
        "pool": pd.read_csv(root / "pool.csv"),
        "reference_path": root / "reference.parquet",
        "threshold": result["metrics"]["threshold"],
        "predict_proba": lambda df: model.predict_proba(df[FEATURE_COLUMNS])[:, 1],
    }


@pytest.fixture
def connection():
    connection = engine.connect()
    transaction = connection.begin()
    connection.execute(delete(OutcomeLabel))
    connection.execute(delete(PredictionEvent))
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _config(trained: dict, tmp_path: Path, **overrides) -> MonitorConfig:
    values = {
        "window_size": 500,
        "min_rows": 300,
        "drift_share_threshold": TEST_DRIFT_THRESHOLD,
        "min_labels": 100,
        "reference_path": trained["reference_path"],
        "reports_dir": tmp_path / "reports",
    }
    values.update(overrides)
    return MonitorConfig(**values)


def _insert_traffic(connection, trained: dict, scenario: str, n_rows: int, seed: int) -> list:
    """Insert scored prediction events for a replayed scenario; returns (id, true_label)."""
    traffic = build_traffic(trained["pool"], scenario, n_rows, np.random.default_rng(seed))
    probabilities = trained["predict_proba"](traffic)
    rows, labels = [], []
    for index, (_, row) in enumerate(traffic.iterrows()):
        prediction_id = uuid4()
        probability = float(probabilities[index])
        rows.append(
            {
                "prediction_id": prediction_id,
                "event_time": BASE_TIME + timedelta(seconds=index),
                "feature_payload": request_body(row),
                "predicted_class": int(probability >= trained["threshold"]),
                "predicted_probability": probability,
                "model_version": "test-version",
                "request_id": f"req-{index}",
                "latency_ms": 1.0,
            }
        )
        labels.append((prediction_id, int(row["Churn"])))
    connection.execute(PredictionEvent.__table__.insert(), rows)
    return labels


def _deliver(connection, labels: list) -> None:
    connection.execute(
        OutcomeLabel.__table__.insert(),
        [
            {
                "prediction_id": prediction_id,
                "observed_label": label,
                "label_time": BASE_TIME + timedelta(hours=1),
                "label_source": "test",
            }
            for prediction_id, label in labels
        ],
    )


def _run_row(connection, run_id):
    table = MonitoringRun.__table__
    return connection.execute(select(table).where(table.c.run_id == run_id)).one()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_baseline_window_has_no_drift(connection, trained: dict, tmp_path: Path) -> None:
    _insert_traffic(connection, trained, "baseline", 500, seed=1)

    result = run_monitoring(connection, _config(trained, tmp_path), trained["predict_proba"])

    assert result.status == "ok"
    assert result.sample_count == 500
    assert result.drift_detected is False
    assert result.drift_share < TEST_DRIFT_THRESHOLD
    assert (tmp_path / "reports" / f"{result.run_id}.html").is_file()
    row = _run_row(connection, result.run_id)
    assert row.status == "ok"
    assert row.drift_detected is False
    assert row.window_start == BASE_TIME
    assert row.window_end == BASE_TIME + timedelta(seconds=499)
    assert row.finished_at >= row.started_at
    assert set(row.drift_summary["features"]) == set(FEATURE_COLUMNS)
    assert "prediction" in row.drift_summary
    assert row.drift_summary["data_quality"]["invalid_data_found"] is False


def test_feature_drift_window_is_detected(connection, trained: dict, tmp_path: Path) -> None:
    _insert_traffic(connection, trained, "feature_drift", 500, seed=1)

    result = run_monitoring(connection, _config(trained, tmp_path), trained["predict_proba"])

    assert result.status == "ok"
    assert result.drift_detected is True
    assert result.drift_share >= TEST_DRIFT_THRESHOLD
    features = result.summary["features"]
    assert features["MonthlyCharges"]["drifted"] is True
    assert features["Contract"]["drifted"] is True
    assert features["gender"]["drifted"] is False
    assert result.summary["data_quality"]["out_of_range"]["MonthlyCharges"][
        "above_reference_max"
    ] > 0


def test_small_window_is_insufficient_data(connection, trained: dict, tmp_path: Path) -> None:
    _insert_traffic(connection, trained, "baseline", 100, seed=2)

    result = run_monitoring(connection, _config(trained, tmp_path), trained["predict_proba"])

    assert result.status == "insufficient_data"
    assert result.sample_count == 100
    assert result.drift_share is None
    assert result.drift_detected is None
    assert result.report_uri is None
    assert not (tmp_path / "reports").exists()
    row = _run_row(connection, result.run_id)
    assert row.status == "insufficient_data"
    assert row.drift_share is None
    assert row.drift_detected is None


def test_few_labels_give_unknown_performance(connection, trained: dict, tmp_path: Path) -> None:
    labels = _insert_traffic(connection, trained, "baseline", 400, seed=3)
    _deliver(connection, labels[:50])

    result = run_monitoring(connection, _config(trained, tmp_path), trained["predict_proba"])

    performance = result.summary["performance"]
    assert performance["status"] == "unknown"
    assert performance["sample_count"] == 50
    assert performance["label_coverage"] == pytest.approx(50 / 400)
    assert performance["f1"] is None
    assert performance["recall"] is None
    assert performance["precision"] is None
    assert result.labeled_count == 50


def test_enough_labels_give_measured_performance(
    connection, trained: dict, tmp_path: Path
) -> None:
    labels = _insert_traffic(connection, trained, "baseline", 400, seed=4)
    _deliver(connection, labels[:250])

    result = run_monitoring(connection, _config(trained, tmp_path), trained["predict_proba"])

    performance = result.summary["performance"]
    assert performance["status"] == "ok"
    assert performance["sample_count"] == 250
    assert 0 < performance["f1"] <= 1
    assert 0 < performance["recall"] <= 1


def test_reference_file_is_unchanged_by_a_run(connection, trained: dict, tmp_path: Path) -> None:
    reference_path: Path = trained["reference_path"]
    before = _sha256(reference_path)
    modified_before = reference_path.stat().st_mtime_ns
    _insert_traffic(connection, trained, "feature_drift", 400, seed=5)

    run_monitoring(connection, _config(trained, tmp_path), trained["predict_proba"])

    assert _sha256(reference_path) == before
    assert reference_path.stat().st_mtime_ns == modified_before


class SavepointEngine:
    """Engine stand-in whose begin() uses a savepoint on the test's rolled-back connection."""

    def __init__(self, connection) -> None:
        self.connection = connection

    @contextmanager
    def begin(self):
        savepoint = self.connection.begin_nested()
        try:
            yield self.connection
        except BaseException:
            savepoint.rollback()
            raise
        else:
            savepoint.commit()


def test_failed_run_is_recorded_and_loop_continues(
    connection, trained: dict, tmp_path: Path
) -> None:
    _insert_traffic(connection, trained, "baseline", 400, seed=6)
    calls = {"count": 0}

    def flaky_predict_proba(df: pd.DataFrame) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("model exploded")
        return trained["predict_proba"](df)

    sleeps: list[float] = []
    results = run_loop(
        SavepointEngine(connection),
        _config(trained, tmp_path),
        flaky_predict_proba,
        interval_seconds=300,
        max_iterations=2,
        sleep=sleeps.append,
    )

    assert [result.status for result in results] == ["failed", "ok"]
    assert sleeps == [300]
    failed = _run_row(connection, results[0].run_id)
    assert failed.status == "failed"
    assert "model exploded" in failed.error
    assert failed.drift_share is None
    assert failed.finished_at is not None
    stored_ok = _run_row(connection, results[1].run_id)
    assert stored_ok.status == "ok"
