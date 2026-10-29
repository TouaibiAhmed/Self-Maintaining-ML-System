"""Utilities for loading and cleaning the raw churn dataset."""

from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a raw churn CSV without modifying its values."""
    return pd.read_csv(Path(path))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn data while preserving rows and customer identifiers."""
    cleaned = df.copy()

    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    cleaned.loc[:, text_columns] = cleaned.loc[:, text_columns].apply(
        lambda column: column.str.strip()
    )

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    if "Churn" in cleaned.columns:
        cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})

    return cleaned
