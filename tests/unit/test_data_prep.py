import pandas as pd

from app.training.data_prep import clean


def test_blank_total_charges_becomes_nan() -> None:
    result = clean(pd.DataFrame({"TotalCharges": [" ", "42.5"]}))

    assert pd.isna(result.loc[0, "TotalCharges"])
    assert result.loc[1, "TotalCharges"] == 42.5


def test_churn_is_mapped_to_binary_values() -> None:
    result = clean(pd.DataFrame({"Churn": [" Yes ", "No"]}))

    assert set(result["Churn"].dropna()) == {0, 1}
    assert result["Churn"].tolist() == [1, 0]


def test_customer_id_is_preserved() -> None:
    result = clean(pd.DataFrame({"customerID": [" abc-123 "], "Churn": ["No"]}))

    assert "customerID" in result.columns
    assert result.loc[0, "customerID"] == "abc-123"
