"""Each SYNTHETIC scenario changes exactly what it claims and nothing else."""

import numpy as np
import pandas as pd
import pytest

from simulator.inject_drift import (
    SCENARIOS,
    apply_scenario,
    baseline,
    concept_drift,
    feature_drift,
    prevalence_shift,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(123)


def test_baseline_leaves_data_unchanged(pool: pd.DataFrame) -> None:
    original = pool.copy()

    result = baseline(pool, _rng())

    pd.testing.assert_frame_equal(result, original)
    assert result is not pool


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_never_modify_their_input(pool: pd.DataFrame, name: str) -> None:
    original = pool.copy()

    apply_scenario(name, pool, _rng())

    pd.testing.assert_frame_equal(pool, original)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_deterministic_for_a_seed(pool: pd.DataFrame, name: str) -> None:
    first = apply_scenario(name, pool, np.random.default_rng(7))
    second = apply_scenario(name, pool, np.random.default_rng(7))

    pd.testing.assert_frame_equal(first, second)


def test_feature_drift_changes_only_charges_and_contract(pool: pd.DataFrame) -> None:
    result = feature_drift(pool, _rng())

    np.testing.assert_allclose(result["MonthlyCharges"], pool["MonthlyCharges"] * 1.30)
    assert result["MonthlyCharges"].mean() == pytest.approx(pool["MonthlyCharges"].mean() * 1.3)

    before = (pool["Contract"] == "Month-to-month").mean()
    after = (result["Contract"] == "Month-to-month").mean()
    # Half of the ~2/3 longer contracts move: the share rises by roughly 1/3.
    assert after - before == pytest.approx((1 - before) * 0.5, abs=0.03)
    was_m2m = pool["Contract"] == "Month-to-month"
    assert (result.loc[was_m2m, "Contract"] == "Month-to-month").all()
    changed = result["Contract"] != pool["Contract"]
    assert (result.loc[changed, "Contract"] == "Month-to-month").all()

    untouched = [c for c in pool.columns if c not in {"MonthlyCharges", "Contract"}]
    pd.testing.assert_frame_equal(result[untouched], pool[untouched])


def test_prevalence_shift_changes_only_the_class_mix(pool: pd.DataFrame) -> None:
    result = prevalence_shift(pool, _rng())

    assert len(result) == len(pool)
    assert pool["Churn"].mean() == pytest.approx(0.265, abs=0.03)
    assert result["Churn"].mean() == pytest.approx(0.45, abs=0.001)
    # Every output row is an exact, unaltered copy of an input row (label included).
    rebuilt = pool.set_index("row_id").loc[result["row_id"]].reset_index()
    pd.testing.assert_frame_equal(result, rebuilt[pool.columns])
    # Within a class, feature distributions are preserved (e.g. churners' mean charges).
    for label in (0, 1):
        assert result.loc[result["Churn"] == label, "MonthlyCharges"].mean() == pytest.approx(
            pool.loc[pool["Churn"] == label, "MonthlyCharges"].mean(), rel=0.05
        )


def test_concept_drift_flips_labels_only_in_the_segment(pool: pd.DataFrame) -> None:
    result = concept_drift(pool, _rng())

    features = [c for c in pool.columns if c != "Churn"]
    pd.testing.assert_frame_equal(result[features], pool[features])

    segment = (pool["Contract"] == "Month-to-month") & (pool["InternetService"] == "Fiber optic")
    flipped = result["Churn"] != pool["Churn"]
    assert not flipped[~segment].any()
    assert flipped[segment].mean() == pytest.approx(0.5, abs=0.06)
    assert set(result["Churn"].unique()) <= {0, 1}


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (feature_drift, {"charges_factor": 0}),
        (feature_drift, {"contract_shift": 1.5}),
        (prevalence_shift, {"target_prevalence": 1.0}),
        (concept_drift, {"flip_probability": -0.1}),
    ],
)
def test_invalid_parameters_are_rejected(pool: pd.DataFrame, func, kwargs) -> None:
    with pytest.raises(ValueError):
        func(pool, _rng(), **kwargs)


def test_prevalence_shift_needs_both_classes(pool: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="each class"):
        prevalence_shift(pool.assign(Churn=0), _rng())


def test_unknown_scenario_is_rejected(pool: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario("everything_drift", pool, _rng())
