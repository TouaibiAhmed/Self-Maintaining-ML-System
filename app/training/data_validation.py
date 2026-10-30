"""Pandera validation for the cleaned Telco churn dataset."""

from __future__ import annotations

import pandas as pd
from pandera import Check, Column, DataFrameSchema

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

NUMERIC_COLUMNS = {
    "SeniorCitizen": int,
    "tenure": int,
    "MonthlyCharges": float,
    "TotalCharges": float,
    "Churn": int,
}


def build_schema(reference: pd.DataFrame) -> DataFrameSchema:
    """Build a schema using categories observed in the cleaned reference data."""
    columns: dict[str, Column] = {}
    for column in REQUIRED_COLUMNS:
        if column in NUMERIC_COLUMNS:
            checks = []
            if column in {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}:
                checks.append(Check.ge(0))
            if column == "Churn":
                checks.append(Check.isin([0, 1]))
            columns[column] = Column(
                NUMERIC_COLUMNS[column],
                checks=checks,
                nullable=column == "TotalCharges",
                required=True,
            )
        else:
            categories = (
                reference[column].dropna().unique().tolist()
                if column in reference.columns
                else []
            )
            columns[column] = Column(
                str,
                checks=Check.isin(categories),
                nullable=False,
                required=True,
            )

    return DataFrameSchema(columns, strict=True, coerce=False, checks=None)


def validate_cleaned(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate cleaned data and return it unchanged when validation succeeds."""
    schema = build_schema(df if reference is None else reference)
    return schema.validate(df, lazy=True)
