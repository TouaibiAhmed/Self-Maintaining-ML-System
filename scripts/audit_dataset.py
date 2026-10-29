"""Audit the raw customer churn dataset and write a Markdown report."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.training.data_prep import clean, load_raw  # noqa: E402

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "dataset_audit.md"


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_table(series: pd.Series) -> str:
    """Render a two-column Markdown table from a Series."""
    lines = ["| Value | Count |", "| --- | ---: |"]
    lines.extend(f"| {value} | {count} |" for value, count in series.items())
    return "\n".join(lines)


def build_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    checksum: str,
    raw_path: Path,
) -> str:
    """Build the dataset audit report."""
    missing = cleaned.isna().sum()
    class_balance = cleaned["Churn"].value_counts(dropna=False).sort_index()
    dtypes = cleaned.dtypes.astype(str)
    columns = "\n".join(f"- `{column}`" for column in raw.columns)
    dtype_table = "\n".join(
        ["| Column | Dtype |", "| --- | --- |"]
        + [f"| `{column}` | `{dtype}` |" for column, dtype in dtypes.items()]
    )
    missing_table = "\n".join(
        ["| Column | Missing values |", "| --- | ---: |"]
        + [f"| `{column}` | {count} |" for column, count in missing.items()]
    )

    return f"""# Dataset Audit

## Source

- File: `{raw_path.relative_to(PROJECT_ROOT)}`
- Rows: {len(raw)}
- Columns: {len(raw.columns)}
- SHA-256: `{checksum}`

## Column names

{columns}

## Dtypes after cleaning

{dtype_table}

## Missing values per column

{missing_table}

Rows containing at least one missing value after cleaning: \
{int(cleaned.isna().any(axis=1).sum())}

## Duplicate rows

Duplicate rows: {int(raw.duplicated().sum())}

## Churn class balance

`Churn=1` is the positive class (churner).

{format_table(class_balance)}

## Cleaning decisions

- Text columns are stripped of leading and trailing whitespace.
- `TotalCharges` is converted with `pd.to_numeric(errors="coerce")`; blank values
  therefore become missing values rather than being silently dropped.
- `Churn` is mapped from `Yes`/`No` to `1`/`0`.
- No rows are dropped during cleaning.
- `customerID` is preserved by cleaning and may be dropped later during training.
"""


def main() -> None:
    """Run the audit, print its facts, and write the Markdown report."""
    raw = load_raw(RAW_PATH)
    cleaned = clean(raw)
    missing = cleaned.isna().sum()
    checksum = sha256(RAW_PATH)

    print("Column names:")
    print("\n".join(f"- {column}" for column in raw.columns))
    print("\nDtypes after cleaning:")
    print(cleaned.dtypes.to_string())
    print("\nMissing values per column:")
    print(missing.to_string())
    print(f"\nRows containing at least one missing value: {int(cleaned.isna().any(axis=1).sum())}")
    print(f"\nDuplicate rows: {int(raw.duplicated().sum())}")
    print("\nClass balance of Churn (1=Yes/churner, 0=No):")
    print(cleaned["Churn"].value_counts(dropna=False).sort_index().to_string())
    print(f"\nSHA-256: {checksum}")

    REPORT_PATH.write_text(
        build_report(raw, cleaned, checksum, RAW_PATH),
        encoding="utf-8",
    )
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
