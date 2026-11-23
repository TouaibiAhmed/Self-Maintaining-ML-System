import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from app.monitoring.labeled_metrics import compute_labeled_metrics


def _frame(predicted: list[int], observed: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"predicted_class": predicted, "observed_label": observed})


def test_below_min_labels_is_unknown_not_zero() -> None:
    result = compute_labeled_metrics(_frame([1, 0, 1], [1, 0, 0]), min_labels=4)

    assert result.status == "unknown"
    assert result.sample_count == 3
    assert result.precision is None
    assert result.recall is None
    assert result.f1 is None


def test_unlabeled_rows_do_not_count_toward_min_labels() -> None:
    df = _frame([1, 1, 0, 0, 1], [1, np.nan, 0, np.nan, np.nan])

    result = compute_labeled_metrics(df, min_labels=3)

    assert result.status == "unknown"
    assert result.sample_count == 2


def test_metrics_use_only_labeled_rows() -> None:
    # The unlabeled rows are all wrong predictions; if they were scored, metrics would drop.
    predicted = [1, 1, 0, 0, 1, 0, 1, 1, 1]
    observed = [1, 0, 0, 1, 1, 0, np.nan, np.nan, np.nan]
    labeled_pred, labeled_true = predicted[:6], [int(v) for v in observed[:6]]

    result = compute_labeled_metrics(_frame(predicted, observed), min_labels=6)

    assert result.status == "ok"
    assert result.sample_count == 6
    assert result.precision == pytest.approx(precision_score(labeled_true, labeled_pred))
    assert result.recall == pytest.approx(recall_score(labeled_true, labeled_pred))
    assert result.f1 == pytest.approx(f1_score(labeled_true, labeled_pred))
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)


def test_exactly_min_labels_is_ok() -> None:
    result = compute_labeled_metrics(_frame([1, 0], [1, 0]), min_labels=2)

    assert result.status == "ok"
    assert result.f1 == pytest.approx(1.0)


def test_undefined_precision_is_none_not_zero() -> None:
    result = compute_labeled_metrics(_frame([0, 0, 0], [1, 0, 0]), min_labels=3)

    assert result.status == "ok"
    assert result.precision is None
    assert result.recall == pytest.approx(0.0)


def test_default_min_labels_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_LABELS", "5")

    result = compute_labeled_metrics(_frame([1, 0, 1], [1, 0, 1]))

    assert result.min_labels == 5
    assert result.status == "unknown"


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_labeled_metrics(pd.DataFrame({"predicted_class": [1]}), min_labels=1)


def test_non_binary_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        compute_labeled_metrics(_frame([1, 0], [2, 0]), min_labels=1)
