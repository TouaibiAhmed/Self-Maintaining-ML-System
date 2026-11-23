"""Model performance computed only on predictions that have a ground-truth label."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

LABEL_COLUMN = "observed_label"
PREDICTION_COLUMN = "predicted_class"
MetricStatus = Literal["ok", "unknown"]


@dataclass(frozen=True)
class LabeledMetrics:
    """Precision, recall and F1 of the churn class (1).

    When ``status`` is "unknown" there are too few labels and every metric is None, never 0.
    A metric can also be None with status "ok" when it is undefined (for example precision
    when the model predicted no churners).
    """

    status: MetricStatus
    sample_count: int
    min_labels: int
    precision: float | None
    recall: float | None
    f1: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_labeled_metrics(df: pd.DataFrame, min_labels: int | None = None) -> LabeledMetrics:
    """Score stored predictions against observed labels.

    ``df`` needs ``predicted_class`` and ``observed_label`` columns. Rows without a label
    (missing ``observed_label``) are ignored, so a window that is only partly labeled can be
    passed directly. ``min_labels`` defaults to the MIN_LABELS setting.
    """
    if min_labels is None:
        from app.core.config import Settings

        min_labels = Settings().min_labels
    if min_labels < 1:
        raise ValueError("min_labels must be >= 1")

    missing = {LABEL_COLUMN, PREDICTION_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labeled = df.dropna(subset=[LABEL_COLUMN])
    sample_count = len(labeled)
    if sample_count < min_labels:
        return LabeledMetrics("unknown", sample_count, min_labels, None, None, None)

    y_true = labeled[LABEL_COLUMN].astype(int)
    y_pred = labeled[PREDICTION_COLUMN].astype(int)
    invalid = set(y_true.unique()) - {0, 1} or set(y_pred.unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Labels and predictions must be 0 or 1, found {sorted(invalid)}")

    def _score(metric) -> float | None:
        value = float(metric(y_true, y_pred, pos_label=1, zero_division=np.nan))
        return None if math.isnan(value) else value

    return LabeledMetrics(
        status="ok",
        sample_count=sample_count,
        min_labels=min_labels,
        precision=_score(precision_score),
        recall=_score(recall_score),
        f1=_score(f1_score),
    )
