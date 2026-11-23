"""Read queries over prediction events and their delayed outcome labels.

A "window" is the DataFrame returned by ``get_recent_window``: one row per prediction_event with
the request features expanded from ``feature_payload`` into columns. Labels are always attached
by joining on prediction_id, never by row order.

All functions take an open SQLAlchemy ``Connection`` so callers control the transaction (tests
use a connection whose transaction is rolled back).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import pandas as pd
from sqlalchemy import Connection, select

from app.core.model_loader import EXPECTED_FEATURE_COLUMNS
from app.db.models import OutcomeLabel, PredictionEvent

EVENT_COLUMNS = [
    "prediction_id",
    "event_time",
    "predicted_class",
    "predicted_probability",
    "model_version",
    "request_id",
    "latency_ms",
]
LABEL_COLUMNS = ["observed_label", "label_time", "label_source"]


@dataclass(frozen=True)
class LabelStats:
    """Label availability for one window. Values are None when they cannot be computed."""

    total_count: int
    labeled_count: int
    label_coverage: float | None
    median_label_delay_seconds: float | None
    latest_label_age_seconds: float | None


def get_recent_window(
    connection: Connection,
    n_rows: int | None = None,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Return the most recent prediction events, oldest first, with features as columns.

    ``n_rows`` keeps only the last N events; ``since`` keeps events at or after that time.
    Both may be combined. With neither, every stored event is returned.
    """
    if n_rows is not None and n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    if since is not None and since.tzinfo is None:
        raise ValueError("since must be timezone-aware")

    table = PredictionEvent.__table__
    query = select(*(table.c[column] for column in EVENT_COLUMNS), table.c.feature_payload)
    query = query.order_by(table.c.event_time.desc(), table.c.prediction_id.desc())
    if since is not None:
        query = query.where(table.c.event_time >= since)
    if n_rows is not None:
        query = query.limit(n_rows)

    rows = list(connection.execute(query).mappings())
    rows.reverse()

    events = pd.DataFrame(
        [{column: row[column] for column in EVENT_COLUMNS} for row in rows],
        columns=EVENT_COLUMNS,
    )
    features = pd.DataFrame.from_records([row["feature_payload"] for row in rows])
    extras = [column for column in features.columns if column not in EXPECTED_FEATURE_COLUMNS]
    features = features.reindex(columns=EXPECTED_FEATURE_COLUMNS + extras)
    window = pd.concat([events, features], axis=1)
    if not window.empty:
        window["prediction_id"] = window["prediction_id"].astype(str)
        window["event_time"] = pd.to_datetime(window["event_time"], utc=True)
    return window


def get_labeled_events(connection: Connection, window: pd.DataFrame) -> pd.DataFrame:
    """Return only the window rows that have an outcome label, with the label columns added.

    The label is found with prediction_events JOIN outcome_labels ON prediction_id.
    """
    empty = window.iloc[0:0].assign(**{column: pd.Series(dtype=object) for column in LABEL_COLUMNS})
    if window.empty:
        return empty

    prediction_ids = [UUID(str(value)) for value in window["prediction_id"]]
    query = (
        select(
            PredictionEvent.prediction_id,
            OutcomeLabel.observed_label,
            OutcomeLabel.label_time,
            OutcomeLabel.label_source,
        )
        .join(OutcomeLabel, OutcomeLabel.prediction_id == PredictionEvent.prediction_id)
        .where(PredictionEvent.prediction_id.in_(prediction_ids))
    )
    labels = pd.DataFrame(
        connection.execute(query).mappings().all(),
        columns=["prediction_id", *LABEL_COLUMNS],
    )
    if labels.empty:
        return empty

    labels["prediction_id"] = labels["prediction_id"].astype(str)
    labels["label_time"] = pd.to_datetime(labels["label_time"], utc=True)
    labeled = window.assign(prediction_id=window["prediction_id"].astype(str)).merge(
        labels, on="prediction_id", how="inner", validate="one_to_one"
    )
    return labeled.reset_index(drop=True)


def compute_label_stats(
    window: pd.DataFrame,
    labeled: pd.DataFrame,
    now: datetime | None = None,
) -> LabelStats:
    """Compute label coverage, median label delay and latest label age (pure function)."""
    now = now or datetime.now(timezone.utc)
    total = len(window)
    labeled_count = len(labeled)
    coverage = labeled_count / total if total else None

    median_delay = None
    latest_age = None
    if labeled_count:
        label_times = pd.to_datetime(labeled["label_time"], utc=True)
        event_times = pd.to_datetime(labeled["event_time"], utc=True)
        median_delay = float((label_times - event_times).dt.total_seconds().median())
        latest_age = float((pd.Timestamp(now) - label_times.max()).total_seconds())

    return LabelStats(
        total_count=total,
        labeled_count=labeled_count,
        label_coverage=coverage,
        median_label_delay_seconds=median_delay,
        latest_label_age_seconds=latest_age,
    )


def get_label_stats(
    connection: Connection,
    window: pd.DataFrame,
    now: datetime | None = None,
) -> LabelStats:
    """Return label coverage (labeled / total), median label delay and latest label age."""
    return compute_label_stats(window, get_labeled_events(connection, window), now=now)
