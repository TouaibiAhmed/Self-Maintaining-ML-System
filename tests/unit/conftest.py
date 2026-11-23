"""Shared fixtures for unit tests."""

import numpy as np
import pandas as pd
import pytest

CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@pytest.fixture
def pool() -> pd.DataFrame:
    """A synthetic pool shaped like production_pool.csv (no real data needed in CI)."""
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({column: rng.choice(values, n) for column, values in CATEGORIES.items()})
    frame.insert(0, "row_id", np.arange(n))
    frame.insert(1, "customerID", [f"C{i:05d}" for i in range(n)])
    frame["SeniorCitizen"] = rng.integers(0, 2, n)
    frame["tenure"] = rng.integers(0, 73, n)
    frame["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    frame["TotalCharges"] = (frame["MonthlyCharges"] * frame["tenure"]).round(2)
    frame.loc[:4, "TotalCharges"] = np.nan
    frame["Churn"] = (rng.random(n) < 0.265).astype(int)
    return frame
