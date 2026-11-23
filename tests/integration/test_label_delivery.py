"""Database tests for label delivery and window queries.

Each test runs inside a transaction that is rolled back, so the development database is left
unchanged. Test events are dated in 2100 so they are always the newest rows in the table.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from app.db.models import OutcomeLabel, PredictionEvent
from app.db.queries import get_label_stats, get_labeled_events, get_recent_window
from app.db.session import engine
from app.monitoring.labeled_metrics import compute_labeled_metrics
from simulator import deliver_labels as deliver_module
from simulator.deliver_labels import deliver_labels
from simulator.pending_labels import append_pending_label

BASE_TIME = datetime(2100, 1, 1, tzinfo=timezone.utc)
FEATURES = {
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


@pytest.fixture
def connection():
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _add_event(connection, minutes_after_base: float, predicted_class: int = 1) -> UUID:
    prediction_id = uuid4()
    connection.execute(
        PredictionEvent.__table__.insert().values(
            prediction_id=prediction_id,
            event_time=BASE_TIME + timedelta(minutes=minutes_after_base),
            feature_payload={**FEATURES, "tenure": int(minutes_after_base)},
            predicted_class=predicted_class,
            predicted_probability=0.8 if predicted_class else 0.2,
            model_version="test-version",
            request_id=f"req-{prediction_id}",
            latency_ms=1.5,
        )
    )
    return prediction_id


def _label_count(connection, prediction_ids) -> int:
    return connection.execute(
        select(func.count())
        .select_from(OutcomeLabel)
        .where(OutcomeLabel.prediction_id.in_(list(prediction_ids)))
    ).scalar_one()


def test_running_delivery_twice_creates_no_duplicates(connection, tmp_path: Path) -> None:
    pending = tmp_path / "pending_labels.csv"
    ids = [_add_event(connection, minutes) for minutes in (0, 1, 2)]
    for prediction_id, label in zip(ids, (1, 0, 1), strict=True):
        append_pending_label(prediction_id, label, path=pending)
    now = BASE_TIME + timedelta(minutes=10)

    first = deliver_labels(connection, pending, min_age_minutes=0, now=now)
    second = deliver_labels(connection, pending, min_age_minutes=0, now=now)

    assert first.delivered == 3
    assert second.delivered == 0
    assert _label_count(connection, ids) == 3
    stored = dict(
        connection.execute(
            select(OutcomeLabel.prediction_id, OutcomeLabel.observed_label).where(
                OutcomeLabel.prediction_id.in_(ids)
            )
        ).all()
    )
    assert stored == dict(zip(ids, (1, 0, 1), strict=True))
    sources = connection.execute(
        select(OutcomeLabel.label_source, OutcomeLabel.label_time).where(
            OutcomeLabel.prediction_id.in_(ids)
        )
    ).all()
    assert {source for source, _ in sources} == {"simulator"}
    assert {label_time for _, label_time in sources} == {now}


def test_existing_label_is_never_overwritten(connection, tmp_path: Path) -> None:
    pending = tmp_path / "pending_labels.csv"
    prediction_id = _add_event(connection, 0)
    append_pending_label(prediction_id, 1, path=pending)
    deliver_labels(connection, pending, now=BASE_TIME + timedelta(minutes=5))

    changed = tmp_path / "changed.csv"
    append_pending_label(prediction_id, 0, path=changed)
    result = deliver_labels(connection, changed, now=BASE_TIME + timedelta(minutes=6))

    assert result.delivered == 0
    label = connection.execute(
        select(OutcomeLabel.observed_label).where(OutcomeLabel.prediction_id == prediction_id)
    ).scalar_one()
    assert label == 1


def test_min_age_max_and_missing_pending_labels(connection, tmp_path: Path) -> None:
    pending = tmp_path / "pending_labels.csv"
    oldest = _add_event(connection, 0)
    middle = _add_event(connection, 5)
    recent = _add_event(connection, 28)
    no_pending = _add_event(connection, 1)
    for prediction_id in (oldest, middle, recent):
        append_pending_label(prediction_id, 1, path=pending)
    now = BASE_TIME + timedelta(minutes=30)

    limited = deliver_labels(connection, pending, min_age_minutes=10, max_labels=1, now=now)
    assert limited.delivered == 1
    assert _label_count(connection, [oldest]) == 1

    rest = deliver_labels(connection, pending, min_age_minutes=10, now=now)
    assert rest.delivered == 1
    assert _label_count(connection, [middle]) == 1
    assert _label_count(connection, [recent, no_pending]) == 0
    assert rest.without_pending_label >= 1


def test_negative_arguments_are_rejected(connection, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        deliver_labels(connection, tmp_path / "x.csv", min_age_minutes=-1)
    with pytest.raises(ValueError):
        deliver_labels(connection, tmp_path / "x.csv", max_labels=0)


def test_cli_rejects_negative_min_age() -> None:
    with pytest.raises(SystemExit):
        deliver_module.main(["--min-age-minutes", "-5"])


def test_recent_window_expands_features_and_keeps_last_n(connection) -> None:
    ids = [_add_event(connection, minutes) for minutes in (0, 1, 2)]

    window = get_recent_window(connection, n_rows=2)

    assert window["prediction_id"].tolist() == [str(ids[1]), str(ids[2])]
    assert window["tenure"].tolist() == [1, 2]
    assert window["Contract"].tolist() == ["Month-to-month"] * 2
    assert "feature_payload" not in window.columns
    assert window["event_time"].is_monotonic_increasing

    since_window = get_recent_window(connection, since=BASE_TIME + timedelta(minutes=1))
    assert since_window["prediction_id"].tolist() == [str(ids[1]), str(ids[2])]


def test_recent_window_rejects_invalid_arguments(connection) -> None:
    with pytest.raises(ValueError):
        get_recent_window(connection, n_rows=0)
    with pytest.raises(ValueError):
        get_recent_window(connection, since=datetime(2100, 1, 1))


def test_labeled_events_stats_and_metrics_use_only_labeled_rows(
    connection, tmp_path: Path
) -> None:
    pending = tmp_path / "pending_labels.csv"
    ids = [_add_event(connection, minutes, predicted_class=1) for minutes in (0, 1, 2, 3)]
    append_pending_label(ids[0], 1, path=pending)
    append_pending_label(ids[1], 0, path=pending)
    label_time = BASE_TIME + timedelta(minutes=10)
    deliver_labels(connection, pending, now=label_time)

    window = get_recent_window(connection, since=BASE_TIME)
    labeled = get_labeled_events(connection, window)
    stats = get_label_stats(connection, window, now=BASE_TIME + timedelta(minutes=20))
    metrics = compute_labeled_metrics(labeled, min_labels=2)

    assert len(window) == 4
    assert sorted(labeled["prediction_id"]) == sorted([str(ids[0]), str(ids[1])])
    assert dict(zip(labeled["prediction_id"], labeled["observed_label"], strict=True)) == {
        str(ids[0]): 1,
        str(ids[1]): 0,
    }
    assert stats.label_coverage == pytest.approx(0.5)
    assert stats.median_label_delay_seconds == pytest.approx(9.5 * 60)
    assert stats.latest_label_age_seconds == pytest.approx(10 * 60)
    assert metrics.status == "ok"
    assert metrics.sample_count == 2
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(1.0)
    assert compute_labeled_metrics(labeled, min_labels=3).status == "unknown"


def test_labeled_events_on_empty_window(connection) -> None:
    window = get_recent_window(connection, since=BASE_TIME + timedelta(days=365))

    labeled = get_labeled_events(connection, window)

    assert window.empty
    assert labeled.empty
    assert "observed_label" in labeled.columns
