"""Unit tests for monitoring data-quality checks, performance status and settings."""

import pandas as pd
import pytest

from app.core.config import Settings
from app.monitoring.drift_report import check_data_quality
from app.monitoring.performance import performance_from_labeled


def _frame(pool: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    return pool.head(n).reset_index(drop=True)


def test_clean_window_has_no_invalid_data(pool: pd.DataFrame) -> None:
    window = _frame(pool).dropna(subset=["TotalCharges"])

    report = check_data_quality(window, pool)

    assert report["missing_columns"] == []
    assert report["unseen_categories"] == {}
    assert report["missing_values"] == {}
    assert report["invalid_data_found"] is False


def test_missing_and_all_null_columns_are_reported(pool: pd.DataFrame) -> None:
    window = _frame(pool).drop(columns=["gender"]).assign(Contract=None)

    report = check_data_quality(window, pool)

    assert report["missing_columns"] == ["gender", "Contract"]
    assert report["invalid_data_found"] is True


def test_unseen_categories_are_reported(pool: pd.DataFrame) -> None:
    window = _frame(pool)
    window.loc[0, "PaymentMethod"] = "Crypto"

    report = check_data_quality(window, pool)

    assert report["unseen_categories"] == {"PaymentMethod": ["Crypto"]}
    assert report["invalid_data_found"] is True


def test_out_of_range_and_negative_values_are_reported(pool: pd.DataFrame) -> None:
    window = _frame(pool)
    window.loc[0, "MonthlyCharges"] = 10_000.0
    window.loc[1, "tenure"] = -3

    report = check_data_quality(window, pool)

    assert report["out_of_range"]["MonthlyCharges"]["above_reference_max"] == 1
    assert report["out_of_range"]["MonthlyCharges"]["negative"] == 0
    assert report["out_of_range"]["tenure"]["negative"] == 1
    assert report["invalid_data_found"] is True


def test_above_reference_max_alone_is_not_invalid(pool: pd.DataFrame) -> None:
    window = _frame(pool).dropna(subset=["TotalCharges"])
    window.loc[window.index[0], "MonthlyCharges"] = 10_000.0

    report = check_data_quality(window, pool)

    assert report["invalid_data_found"] is False


def test_missing_values_are_counted(pool: pd.DataFrame) -> None:
    report = check_data_quality(_frame(pool), pool)

    assert report["missing_values"] == {"TotalCharges": 5}


def test_performance_below_min_labels_is_unknown_with_null_metrics() -> None:
    window = pd.DataFrame({"prediction_id": list("abcd")})
    labeled = pd.DataFrame({"predicted_class": [1, 0], "observed_label": [1, 0]})

    result = performance_from_labeled(window, labeled, min_labels=3)

    assert result.status == "unknown"
    assert (result.precision, result.recall, result.f1) == (None, None, None)
    assert result.sample_count == 2
    assert result.label_coverage == pytest.approx(0.5)


def test_min_rows_above_window_size_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONITOR_WINDOW_SIZE", "100")
    monkeypatch.setenv("MONITOR_MIN_ROWS", "300")

    with pytest.raises(ValueError, match="MONITOR_MIN_ROWS"):
        Settings()
