"""Data-quality checks and Evidently drift reports for one monitoring window.

Evidently version and API style: evidently==0.7.23, using the current (0.7+) API:
``evidently.Report`` with ``evidently.presets.DataDriftPreset`` and
``evidently.metrics.ValueDrift``,
inputs wrapped with ``evidently.Dataset.from_pandas(df, data_definition=DataDefinition(...))``, and
``Report.run(current_data, reference_data)`` returning a ``Run`` that has ``.dict()`` and
``.save_html()``. This is NOT the legacy ``evidently.report.Report`` / ``ColumnMapping`` API of
Evidently < 0.7. Names were checked against the installed source (evidently/presets/drift.py and
evidently/metrics/column_statistics.py).

Stat tests are pinned rather than auto-selected, because Evidently's automatic choice depends on
the reference size (p-value tests below 1000 rows, distances above). Both pinned methods are
distances, so a column drifts when its score is >= its threshold:
- numeric features and the prediction probability: normed Wasserstein distance, threshold 0.1
- categorical features: Jensen-Shannon distance, threshold 0.1

Drift means the input data changed. It does not prove the model got worse; only labels can.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from app.core.model_loader import EXPECTED_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference.parquet"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports" / "monitoring"

FEATURE_COLUMNS = list(EXPECTED_FEATURE_COLUMNS)
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [c for c in FEATURE_COLUMNS if c not in NUMERIC_FEATURES]
PREDICTION_COLUMN = "predicted_probability"

NUMERIC_DRIFT_METHOD = "wasserstein"
CATEGORICAL_DRIFT_METHOD = "jensenshannon"
COLUMN_DRIFT_THRESHOLD = 0.1


@dataclass
class DriftResult:
    """Feature and prediction drift for one window, plus the Evidently run for the HTML."""

    drift_share: float
    drifted_count: int
    features: dict[str, dict[str, Any]]
    prediction: dict[str, Any]
    run: Any


def load_reference(path: Path = DEFAULT_REFERENCE_PATH) -> pd.DataFrame:
    """Read the fixed reference sample. This module never writes to it."""
    reference = pd.read_parquet(path)
    missing = [column for column in FEATURE_COLUMNS if column not in reference.columns]
    if missing:
        raise ValueError(f"Reference file {path} is missing feature columns: {missing}")
    if reference.empty:
        raise ValueError(f"Reference file {path} has no rows")
    return reference


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature columns with numeric dtypes coerced (JSON nulls become NaN)."""
    prepared = df.reindex(columns=FEATURE_COLUMNS).copy()
    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def check_data_quality(window: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Check the window for missing columns, unseen categories, bad ranges and missing values.

    - missing_columns: feature columns absent from the window or entirely null
    - unseen_categories: categorical values never seen in the reference
    - out_of_range: per numeric column, values below 0 (invalid) and outside the reference
      [min, max] (informational: a shift such as higher charges shows up here)
    - missing_values: null count per column (TotalCharges is legitimately null for new customers)
    """
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in window.columns or window[column].isna().all()
    ]
    present = [column for column in FEATURE_COLUMNS if column not in missing_columns]

    unseen_categories: dict[str, list[str]] = {}
    for column in CATEGORICAL_FEATURES:
        if column not in present:
            continue
        known = set(reference[column].dropna().astype(str))
        unseen = sorted(set(window[column].dropna().astype(str)) - known)
        if unseen:
            unseen_categories[column] = unseen

    out_of_range: dict[str, dict[str, Any]] = {}
    for column in NUMERIC_FEATURES:
        if column not in present:
            continue
        values = pd.to_numeric(window[column], errors="coerce")
        ref_values = pd.to_numeric(reference[column], errors="coerce")
        ref_min, ref_max = float(ref_values.min()), float(ref_values.max())
        counts = {
            "negative": int((values < 0).sum()),
            "below_reference_min": int((values < ref_min).sum()),
            "above_reference_max": int((values > ref_max).sum()),
        }
        if any(counts.values()):
            out_of_range[column] = {**counts, "reference_min": ref_min, "reference_max": ref_max}

    missing_values = {
        column: int(window[column].isna().sum())
        for column in present
        if int(window[column].isna().sum()) > 0
    }
    invalid = bool(
        missing_columns
        or unseen_categories
        or any(entry["negative"] for entry in out_of_range.values())
    )
    return {
        "missing_columns": missing_columns,
        "unseen_categories": unseen_categories,
        "out_of_range": out_of_range,
        "missing_values": missing_values,
        "invalid_data_found": invalid,
    }


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_share_threshold: float,
) -> DriftResult:
    """Compare current features and predicted probabilities with the reference.

    Both frames need the model feature columns and ``predicted_probability``. Feature drift
    (DataDriftPreset over the 19 features) gives drift_share; prediction drift is a separate
    ValueDrift metric and is NOT counted in drift_share.
    """
    columns = [*FEATURE_COLUMNS, PREDICTION_COLUMN]
    reference_data = prepare_features(reference).assign(
        **{PREDICTION_COLUMN: pd.to_numeric(reference[PREDICTION_COLUMN])}
    )
    current_data = prepare_features(current).assign(
        **{PREDICTION_COLUMN: pd.to_numeric(current[PREDICTION_COLUMN])}
    )
    definition = DataDefinition(
        numerical_columns=[*NUMERIC_FEATURES, PREDICTION_COLUMN],
        categorical_columns=CATEGORICAL_FEATURES,
    )
    report = Report(
        [
            DataDriftPreset(
                columns=FEATURE_COLUMNS,
                drift_share=drift_share_threshold,
                num_method=NUMERIC_DRIFT_METHOD,
                cat_method=CATEGORICAL_DRIFT_METHOD,
                threshold=COLUMN_DRIFT_THRESHOLD,
            ),
            ValueDrift(
                column=PREDICTION_COLUMN,
                method=NUMERIC_DRIFT_METHOD,
                threshold=COLUMN_DRIFT_THRESHOLD,
            ),
        ]
    )
    run = report.run(
        Dataset.from_pandas(current_data[columns], data_definition=definition),
        Dataset.from_pandas(reference_data[columns], data_definition=definition),
    )
    return _parse_run(run, drift_share_threshold)


def _parse_run(run: Any, drift_share_threshold: float) -> DriftResult:
    features: dict[str, dict[str, Any]] = {}
    prediction: dict[str, Any] | None = None
    evidently_count: dict[str, float] | None = None
    for metric in run.dict()["metrics"]:
        config = metric["config"]
        metric_type = str(config.get("type", ""))
        if metric_type.endswith("DriftedColumnsCount"):
            evidently_count = metric["value"]
        elif metric_type.endswith("ValueDrift"):
            score = float(metric["value"])
            threshold = float(config["threshold"])
            entry = {
                "method": config["method"],
                "score": score,
                "threshold": threshold,
                "drifted": score >= threshold,
            }
            if config["column"] == PREDICTION_COLUMN:
                prediction = entry
            else:
                features[config["column"]] = entry

    if evidently_count is None or prediction is None or set(features) != set(FEATURE_COLUMNS):
        raise RuntimeError("Evidently output is missing expected drift metrics")
    drifted_count = sum(1 for entry in features.values() if entry["drifted"])
    if drifted_count != int(evidently_count["count"]):
        raise RuntimeError(
            f"Drifted column count mismatch: computed {drifted_count}, "
            f"Evidently reported {evidently_count['count']}"
        )
    return DriftResult(
        drift_share=float(evidently_count["share"]),
        drifted_count=drifted_count,
        features=features,
        prediction=prediction,
        run=run,
    )


def save_html_report(run: Any, reports_dir: Path, run_id: str) -> Path:
    """Write the Evidently HTML report to ``<reports_dir>/<run_id>.html``."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{run_id}.html"
    run.save_html(str(path))
    return path
