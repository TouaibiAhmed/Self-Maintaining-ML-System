"""Monitoring job: drift, data quality and labeled performance for the recent window.

Usage (from the project root):
    python -m app.monitoring.run_monitor --once
    python -m app.monitoring.run_monitor --loop      # every MONITOR_INTERVAL_SECONDS

Each run writes exactly one monitoring_runs row with status "ok", "insufficient_data" or
"failed". A failed run records the error and, in --loop mode, the loop keeps going.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine

from app.db.models import MonitoringRun
from app.db.queries import get_recent_window
from app.monitoring.drift_report import (
    DEFAULT_REFERENCE_PATH,
    DEFAULT_REPORTS_DIR,
    FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    PROJECT_ROOT,
    check_data_quality,
    load_reference,
    prepare_features,
    run_drift_report,
    save_html_report,
)
from app.monitoring.performance import compute_window_performance

logger = logging.getLogger(__name__)

PredictProba = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class MonitorConfig:
    """Thresholds and paths for one monitoring run."""

    window_size: int
    min_rows: int
    drift_share_threshold: float
    min_labels: int
    reference_path: Path = DEFAULT_REFERENCE_PATH
    reports_dir: Path = DEFAULT_REPORTS_DIR

    @classmethod
    def from_settings(cls, settings: Any) -> MonitorConfig:
        return cls(
            window_size=settings.monitor_window_size,
            min_rows=settings.monitor_min_rows,
            drift_share_threshold=settings.drift_share_threshold,
            min_labels=settings.min_labels,
        )


@dataclass
class MonitorResult:
    """What one monitoring run measured; mirrors the stored monitoring_runs row."""

    run_id: UUID
    status: str
    sample_count: int = 0
    labeled_count: int = 0
    drift_share: float | None = None
    drift_detected: bool | None = None
    report_uri: str | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _report_uri(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _config_summary(config: MonitorConfig) -> dict[str, Any]:
    return {
        "window_size": config.window_size,
        "min_rows": config.min_rows,
        "drift_share_threshold": config.drift_share_threshold,
        "min_labels": config.min_labels,
    }


def _store(connection: Connection, result: MonitorResult, started_at: datetime, window) -> None:
    has_rows = window is not None and not window.empty
    connection.execute(
        MonitoringRun.__table__.insert().values(
            run_id=result.run_id,
            started_at=started_at,
            finished_at=_now(),
            window_start=window["event_time"].min().to_pydatetime() if has_rows else None,
            window_end=window["event_time"].max().to_pydatetime() if has_rows else None,
            sample_count=result.sample_count,
            labeled_count=result.labeled_count,
            drift_share=result.drift_share,
            drift_detected=result.drift_detected,
            drift_summary=result.summary,
            report_uri=result.report_uri,
            status=result.status,
            error=result.error,
        )
    )


def run_monitoring(
    connection: Connection,
    config: MonitorConfig,
    predict_proba: PredictProba,
    run_id: UUID | None = None,
) -> MonitorResult:
    """Run one monitoring pass on ``connection`` and store its monitoring_runs row.

    ``predict_proba`` scores the reference features with the active model so the reference
    probability distribution matches the model currently serving. The reference file is only
    read, never written.
    """
    run_id = run_id or uuid4()
    started_at = _now()
    window = get_recent_window(connection, n_rows=config.window_size)
    performance = compute_window_performance(connection, window, config.min_labels)
    result = MonitorResult(
        run_id=run_id,
        status="ok",
        sample_count=len(window),
        labeled_count=performance.sample_count,
        summary={"config": _config_summary(config), "performance": performance.as_dict()},
    )

    if len(window) < config.min_rows:
        result.status = "insufficient_data"
        result.summary["reason"] = (
            f"window has {len(window)} rows; at least {config.min_rows} are required"
        )
        _store(connection, result, started_at, window)
        return result

    reference = load_reference(config.reference_path)
    reference_scored = prepare_features(reference)
    reference_scored[PREDICTION_COLUMN] = np.asarray(predict_proba(reference_scored), dtype=float)

    quality = check_data_quality(window, reference)
    drift = run_drift_report(
        reference_scored,
        window[[*FEATURE_COLUMNS, PREDICTION_COLUMN]],
        config.drift_share_threshold,
    )
    report_path = save_html_report(drift.run, config.reports_dir, str(run_id))

    result.drift_share = drift.drift_share
    result.drift_detected = drift.drift_share >= config.drift_share_threshold
    result.report_uri = _report_uri(report_path)
    result.summary.update(
        {
            "drifted_feature_count": drift.drifted_count,
            "feature_count": len(drift.features),
            "features": drift.features,
            "prediction": drift.prediction,
            "data_quality": quality,
            "reference_rows": len(reference),
        }
    )
    _store(connection, result, started_at, window)
    return result


def record_failed_run(
    connection: Connection, run_id: UUID, started_at: datetime, error: str
) -> MonitorResult:
    """Store a monitoring_runs row with status "failed" and the error message."""
    result = MonitorResult(run_id=run_id, status="failed", error=error[:2000])
    _store(connection, result, started_at, None)
    return result


def run_once(engine: Engine, config: MonitorConfig, predict_proba: PredictProba) -> MonitorResult:
    """Run monitoring in its own transaction; on any error record a failed run instead."""
    run_id = uuid4()
    started_at = _now()
    try:
        with engine.begin() as connection:
            return run_monitoring(connection, config, predict_proba, run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded, not raised
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("monitoring_run_failed", extra={"event": "monitoring_run_failed"})
        try:
            with engine.begin() as connection:
                return record_failed_run(connection, run_id, started_at, error)
        except Exception:  # noqa: BLE001 - e.g. the database itself is down
            logger.exception(
                "monitoring_failure_not_recorded",
                extra={"event": "monitoring_failure_not_recorded"},
            )
            return MonitorResult(run_id=run_id, status="failed", error=error)


def run_loop(
    engine: Engine,
    config: MonitorConfig,
    predict_proba: PredictProba,
    interval_seconds: float,
    max_iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_result: Callable[[MonitorResult], None] | None = None,
) -> list[MonitorResult]:
    """Run monitoring every ``interval_seconds``; a failed run never stops the loop."""
    results: list[MonitorResult] = []
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        result = run_once(engine, config, predict_proba)
        results.append(result)
        if on_result is not None:
            on_result(result)
        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            sleep(interval_seconds)
    return results


def format_result(result: MonitorResult) -> str:
    """One human-readable summary block for the console."""
    performance = result.summary.get("performance", {})
    lines = [
        f"run_id={result.run_id} status={result.status}",
        f"  sample_count={result.sample_count} labeled_count={result.labeled_count}",
    ]
    if result.drift_share is not None:
        drifted = [
            name for name, entry in result.summary.get("features", {}).items() if entry["drifted"]
        ]
        lines.append(
            f"  drift_share={result.drift_share:.3f} drift_detected={result.drift_detected} "
            f"drifted_features={drifted}"
        )
        prediction = result.summary.get("prediction", {})
        if prediction:
            lines.append(
                f"  prediction_drift_score={prediction['score']:.3f} "
                f"prediction_drifted={prediction['drifted']}"
            )
        lines.append(f"  report={result.report_uri}")
    if performance:
        lines.append(
            f"  performance status={performance['status']} f1={performance['f1']} "
            f"recall={performance['recall']} precision={performance['precision']} "
            f"labeled={performance['sample_count']} coverage={performance['label_coverage']}"
        )
    if result.error:
        lines.append(f"  error={result.error}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the drift and performance monitoring job.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one monitoring pass and exit.")
    mode.add_argument("--loop", action="store_true", help="Run every MONITOR_INTERVAL_SECONDS.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from app.core.config import Settings
    from app.core.logging import configure_logging
    from app.core.model_loader import ModelStore
    from app.db.session import create_db_engine

    settings = Settings()
    configure_logging(settings.log_level)
    config = MonitorConfig.from_settings(settings)
    model = ModelStore(settings.model_dir)

    def predict_proba(features: pd.DataFrame) -> np.ndarray:
        return model.active_model.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    engine = create_db_engine(settings)
    try:
        if args.once:
            result = run_once(engine, config, predict_proba)
            print(format_result(result))
            return 1 if result.status == "failed" else 0
        run_loop(
            engine,
            config,
            predict_proba,
            settings.monitor_interval_seconds,
            on_result=lambda result: print(format_result(result), flush=True),
        )
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
