"""SYNTHETIC drift scenarios for the traffic simulator (spec section 6.3).

Every function here takes a DataFrame of production-pool rows (features plus the true ``Churn``
label) and returns a NEW DataFrame; the input is never modified. Each scenario changes exactly one
kind of thing so experiments can tell them apart:

| scenario          | what changes                          | labels                       |
| ----------------- | ------------------------------------- | ---------------------------- |
| baseline          | nothing                               | unchanged                    |
| feature_drift     | MonthlyCharges +30%, more M2M         | unchanged                    |
| prevalence_shift  | class mix resampled to ~45% churn     | unchanged per row            |
| concept_drift     | nothing in the features               | flipped in one segment       |

These are synthetic transformations of a public dataset. They are NOT real customer behavior and
no business conclusion may be drawn from them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

LABEL_COLUMN = "Churn"
MONTH_TO_MONTH = "Month-to-month"
FIBER_OPTIC = "Fiber optic"

FEATURE_DRIFT_CHARGES_FACTOR = 1.30
FEATURE_DRIFT_CONTRACT_SHIFT = 0.50
PREVALENCE_TARGET = 0.45
CONCEPT_DRIFT_FLIP_PROBABILITY = 0.50

Scenario = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


def baseline(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """SYNTHETIC baseline: return the rows unchanged.

    Demonstrates: normal service behavior and the metrics to expect when live traffic has the
    same distribution as the training/reference data.
    Does NOT demonstrate: production readiness; it only replays held-out public data.
    """
    # SYNTHETIC: identity transformation; the copy guarantees the caller's frame is untouched.
    return df.copy()


def feature_drift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    charges_factor: float = FEATURE_DRIFT_CHARGES_FACTOR,
    contract_shift: float = FEATURE_DRIFT_CONTRACT_SHIFT,
) -> pd.DataFrame:
    """SYNTHETIC covariate (feature) drift: inputs change, labels do not.

    MonthlyCharges is multiplied by ``charges_factor`` (+30% by default) and each customer on a
    One year or Two year contract is moved to Month-to-month with probability
    ``contract_shift``. TotalCharges and every other column, including the Churn label, are left
    exactly as they were.

    Demonstrates: that the monitoring job detects a shift in input distributions.
    Does NOT demonstrate: that model accuracy fell. The labels were not changed to follow the new
    features, so any change in labeled performance must be measured, never assumed.
    """
    # SYNTHETIC: a distribution shift in two inputs only; not a model of real customer behavior.
    if charges_factor <= 0:
        raise ValueError("charges_factor must be > 0")
    if not 0 <= contract_shift <= 1:
        raise ValueError("contract_shift must be between 0 and 1")

    drifted = df.copy()
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * charges_factor
    longer_contract = (drifted["Contract"] != MONTH_TO_MONTH).to_numpy()
    moved = longer_contract & (rng.random(len(drifted)) < contract_shift)
    drifted.loc[moved, "Contract"] = MONTH_TO_MONTH
    return drifted


def prevalence_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_prevalence: float = PREVALENCE_TARGET,
) -> pd.DataFrame:
    """SYNTHETIC label (prevalence) shift: churners become more common.

    Rows are resampled with replacement so that about ``target_prevalence`` of them are churners
    (45% by default; the pool has about 26.5%). The output has the same number of rows. Every
    output row is an unmodified copy of an input row, so the feature distribution WITHIN each
    class and the feature-label relationship stay the same; only the class mix changes.

    Demonstrates: monitoring of the prediction and label distributions when prevalence moves.
    Does NOT demonstrate: concept drift. It may call for threshold and calibration analysis, but
    it does not by itself mean the model is wrong.
    """
    # SYNTHETIC: class-conditional resampling of existing rows; no row is altered.
    if not 0 < target_prevalence < 1:
        raise ValueError("target_prevalence must be between 0 and 1")
    positives = np.flatnonzero(df[LABEL_COLUMN].to_numpy() == 1)
    negatives = np.flatnonzero(df[LABEL_COLUMN].to_numpy() == 0)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("prevalence_shift needs at least one row of each class")

    n_rows = len(df)
    n_positive = int(round(n_rows * target_prevalence))
    chosen = np.concatenate(
        [
            rng.choice(positives, size=n_positive, replace=True),
            rng.choice(negatives, size=n_rows - n_positive, replace=True),
        ]
    )
    rng.shuffle(chosen)
    return df.iloc[chosen].reset_index(drop=True)


def concept_drift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    flip_probability: float = CONCEPT_DRIFT_FLIP_PROBABILITY,
) -> pd.DataFrame:
    """SYNTHETIC concept drift: the feature-label relationship changes in one segment.

    For customers with Contract = Month-to-month AND InternetService = Fiber optic, the true
    Churn label is flipped with probability ``flip_probability`` (0.5 by default). Features are
    not changed at all, so input-drift detectors should see nothing unusual.

    Demonstrates: a possible drop in labeled performance that only delayed labels can reveal.
    Does NOT demonstrate: real customer behavior. A random label flip is an artificial stand-in
    for a change in why customers churn.
    """
    # SYNTHETIC: label noise injected into one segment; every feature column is left untouched.
    if not 0 <= flip_probability <= 1:
        raise ValueError("flip_probability must be between 0 and 1")

    drifted = df.copy()
    segment = (
        (drifted["Contract"] == MONTH_TO_MONTH) & (drifted["InternetService"] == FIBER_OPTIC)
    ).to_numpy()
    flipped = segment & (rng.random(len(drifted)) < flip_probability)
    drifted.loc[flipped, LABEL_COLUMN] = 1 - drifted.loc[flipped, LABEL_COLUMN]
    return drifted


SCENARIOS: dict[str, Scenario] = {
    "baseline": baseline,
    "feature_drift": feature_drift,
    "prevalence_shift": prevalence_shift,
    "concept_drift": concept_drift,
}


def apply_scenario(name: str, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply a named scenario with its default parameters."""
    try:
        scenario = SCENARIOS[name]
    except KeyError:
        raise ValueError(f"Unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from None
    return scenario(df, rng)
