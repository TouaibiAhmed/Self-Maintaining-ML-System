from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app.db.queries import compute_label_stats

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)


def test_coverage_delay_and_latest_age_on_hand_made_window() -> None:
    event_times = [NOW - timedelta(hours=h) for h in (4, 3, 2, 1)]
    window = pd.DataFrame({"prediction_id": ["a", "b", "c", "d"], "event_time": event_times})
    # Delays: a = 60 s, b = 300 s, c = 120 s; d is unlabeled.
    labeled = pd.DataFrame(
        {
            "prediction_id": ["a", "b", "c"],
            "event_time": event_times[:3],
            "label_time": [
                event_times[0] + timedelta(seconds=60),
                event_times[1] + timedelta(seconds=300),
                event_times[2] + timedelta(seconds=120),
            ],
        }
    )

    stats = compute_label_stats(window, labeled, now=NOW)

    assert stats.total_count == 4
    assert stats.labeled_count == 3
    assert stats.label_coverage == pytest.approx(0.75)
    assert stats.median_label_delay_seconds == pytest.approx(120.0)
    # Latest label is c: 2 h before NOW plus 120 s.
    assert stats.latest_label_age_seconds == pytest.approx(2 * 3600 - 120)


def test_window_without_labels_has_zero_coverage_and_no_delay() -> None:
    window = pd.DataFrame({"prediction_id": ["a"], "event_time": [NOW]})
    labeled = pd.DataFrame(columns=["prediction_id", "event_time", "label_time"])

    stats = compute_label_stats(window, labeled, now=NOW)

    assert stats.label_coverage == 0.0
    assert stats.median_label_delay_seconds is None
    assert stats.latest_label_age_seconds is None


def test_empty_window_has_unknown_coverage() -> None:
    empty = pd.DataFrame(columns=["prediction_id", "event_time", "label_time"])

    stats = compute_label_stats(empty, empty, now=NOW)

    assert stats.total_count == 0
    assert stats.label_coverage is None
