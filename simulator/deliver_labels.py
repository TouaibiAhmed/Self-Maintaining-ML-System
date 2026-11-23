"""Deliver simulated delayed ground-truth labels to stored predictions.

Usage (from the project root):
    python simulator/deliver_labels.py --min-age-minutes 0
    python simulator/deliver_labels.py --min-age-minutes 30 --max 200

Selects prediction_events older than --min-age-minutes that have no outcome_labels row, looks up
their true label in data/processed/pending_labels.csv by prediction_id, and inserts them with
INSERT ... ON CONFLICT DO NOTHING, so re-running never creates duplicates or overwrites a label.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python simulator/deliver_labels.py`: put the project root on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import Connection, func, select  # noqa: E402
from sqlalchemy.dialects.postgresql import insert  # noqa: E402

from app.db.models import OutcomeLabel, PredictionEvent  # noqa: E402
from simulator.pending_labels import (  # noqa: E402
    DEFAULT_PENDING_LABELS_PATH,
    load_pending_labels,
)

LABEL_SOURCE = "simulator"


@dataclass(frozen=True)
class DeliveryResult:
    """Counts from one label delivery run."""

    delivered: int
    eligible_unlabeled: int
    without_pending_label: int
    invalid_pending_rows: int
    conflicting_pending_ids: int


def deliver_labels(
    connection: Connection,
    pending_labels_path: Path = DEFAULT_PENDING_LABELS_PATH,
    min_age_minutes: float = 0,
    max_labels: int | None = None,
    now: datetime | None = None,
) -> DeliveryResult:
    """Insert outcome labels for old-enough, unlabeled predictions.

    The caller owns the transaction (for example ``with engine.begin() as connection``), so a
    failure part-way through rolls back every insert of this run.
    """
    if min_age_minutes < 0:
        raise ValueError("min_age_minutes must be >= 0")
    if max_labels is not None and max_labels < 1:
        raise ValueError("max_labels must be >= 1")

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=min_age_minutes)
    pending = load_pending_labels(pending_labels_path)

    unlabeled_old = (
        select(PredictionEvent.prediction_id)
        .outerjoin(OutcomeLabel, OutcomeLabel.prediction_id == PredictionEvent.prediction_id)
        .where(OutcomeLabel.prediction_id.is_(None))
        .where(PredictionEvent.event_time <= cutoff)
    )
    eligible_unlabeled = connection.execute(
        select(func.count()).select_from(unlabeled_old.subquery())
    ).scalar_one()

    matched: list = []
    if pending.labels:
        query = unlabeled_old.where(
            PredictionEvent.prediction_id.in_(list(pending.labels))
        ).order_by(PredictionEvent.event_time, PredictionEvent.prediction_id)
        if max_labels is not None:
            query = query.limit(max_labels)
        matched = list(connection.execute(query).scalars())

    delivered = 0
    if matched:
        statement = (
            insert(OutcomeLabel)
            .values(
                [
                    {
                        "prediction_id": prediction_id,
                        "observed_label": pending.labels[prediction_id],
                        "label_time": now,
                        "label_source": LABEL_SOURCE,
                    }
                    for prediction_id in matched
                ]
            )
            .on_conflict_do_nothing(index_elements=[OutcomeLabel.prediction_id])
            .returning(OutcomeLabel.prediction_id)
        )
        delivered = len(connection.execute(statement).all())

    return DeliveryResult(
        delivered=delivered,
        eligible_unlabeled=eligible_unlabeled,
        without_pending_label=_count_without_pending(connection, unlabeled_old, pending.labels),
        invalid_pending_rows=pending.invalid_rows,
        conflicting_pending_ids=len(pending.conflicting_ids),
    )


def _count_without_pending(connection: Connection, unlabeled_old, labels: dict) -> int:
    """Count eligible predictions (before this run's inserts) that have no pending label."""
    query = unlabeled_old
    if labels:
        query = query.where(PredictionEvent.prediction_id.not_in(list(labels)))
    return connection.execute(select(func.count()).select_from(query.subquery())).scalar_one()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--min-age-minutes",
        type=float,
        default=0,
        help="Only label predictions at least this many minutes old (default: 0).",
    )
    parser.add_argument(
        "--max",
        dest="max_labels",
        type=int,
        default=None,
        help="Deliver at most this many labels in this run (default: no limit).",
    )
    parser.add_argument(
        "--pending-labels",
        type=Path,
        default=DEFAULT_PENDING_LABELS_PATH,
        help="CSV of (prediction_id, true_label) pairs.",
    )
    args = parser.parse_args(argv)
    if args.min_age_minutes < 0:
        parser.error("--min-age-minutes must be >= 0")
    if args.max_labels is not None and args.max_labels < 1:
        parser.error("--max must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from app.core.config import Settings
    from app.db.session import create_db_engine

    engine = create_db_engine(Settings())
    try:
        with engine.begin() as connection:
            result = deliver_labels(
                connection,
                pending_labels_path=args.pending_labels,
                min_age_minutes=args.min_age_minutes,
                max_labels=args.max_labels,
            )
    finally:
        engine.dispose()

    print(f"Delivered {result.delivered} labels.")
    print(
        f"Eligible unlabeled predictions: {result.eligible_unlabeled}; "
        f"without a pending label: {result.without_pending_label}."
    )
    if result.invalid_pending_rows or result.conflicting_pending_ids:
        print(
            f"Skipped {result.invalid_pending_rows} invalid rows and "
            f"{result.conflicting_pending_ids} prediction_ids with conflicting labels "
            "in the pending labels file."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
