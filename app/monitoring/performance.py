"""Labeled model performance for one monitoring window.

Only rows with a delivered ground-truth label are scored. Below MIN_LABELS labeled rows the
status is "unknown" and precision, recall and F1 are None: never 0 and never "healthy".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd
from sqlalchemy import Connection

from app.db.queries import get_labeled_events
from app.monitoring.labeled_metrics import compute_labeled_metrics


@dataclass(frozen=True)
class WindowPerformance:
    """Precision, recall and F1 on the labeled rows of a window, with sample sizes."""

    status: Literal["ok", "unknown"]
    sample_count: int
    total_count: int
    label_coverage: float | None
    min_labels: int
    precision: float | None
    recall: float | None
    f1: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def performance_from_labeled(
    window: pd.DataFrame, labeled: pd.DataFrame, min_labels: int
) -> WindowPerformance:
    """Compute window performance from an already-joined labeled frame (pure function)."""
    metrics = compute_labeled_metrics(labeled, min_labels=min_labels)
    total = len(window)
    return WindowPerformance(
        status=metrics.status,
        sample_count=metrics.sample_count,
        total_count=total,
        label_coverage=metrics.sample_count / total if total else None,
        min_labels=min_labels,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
    )


def compute_window_performance(
    connection: Connection, window: pd.DataFrame, min_labels: int
) -> WindowPerformance:
    """Join the window with outcome_labels on prediction_id and score the labeled rows."""
    return performance_from_labeled(window, get_labeled_events(connection, window), min_labels)
