# Dataset Audit

## Source

- File: `data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Rows: 7043
- Columns: 21
- SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`

## Column names

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

## Dtypes after cleaning

| Column | Dtype |
| --- | --- |
| `customerID` | `object` |
| `gender` | `object` |
| `SeniorCitizen` | `int64` |
| `Partner` | `object` |
| `Dependents` | `object` |
| `tenure` | `int64` |
| `PhoneService` | `object` |
| `MultipleLines` | `object` |
| `InternetService` | `object` |
| `OnlineSecurity` | `object` |
| `OnlineBackup` | `object` |
| `DeviceProtection` | `object` |
| `TechSupport` | `object` |
| `StreamingTV` | `object` |
| `StreamingMovies` | `object` |
| `Contract` | `object` |
| `PaperlessBilling` | `object` |
| `PaymentMethod` | `object` |
| `MonthlyCharges` | `float64` |
| `TotalCharges` | `float64` |
| `Churn` | `int64` |

## Missing values per column

| Column | Missing values |
| --- | ---: |
| `customerID` | 0 |
| `gender` | 0 |
| `SeniorCitizen` | 0 |
| `Partner` | 0 |
| `Dependents` | 0 |
| `tenure` | 0 |
| `PhoneService` | 0 |
| `MultipleLines` | 0 |
| `InternetService` | 0 |
| `OnlineSecurity` | 0 |
| `OnlineBackup` | 0 |
| `DeviceProtection` | 0 |
| `TechSupport` | 0 |
| `StreamingTV` | 0 |
| `StreamingMovies` | 0 |
| `Contract` | 0 |
| `PaperlessBilling` | 0 |
| `PaymentMethod` | 0 |
| `MonthlyCharges` | 0 |
| `TotalCharges` | 11 |
| `Churn` | 0 |

Rows containing at least one missing value after cleaning: 11

## Duplicate rows

Duplicate rows: 0

## Churn class balance

`Churn=1` is the positive class (churner).

| Value | Count |
| --- | ---: |
| 0 | 5174 |
| 1 | 1869 |

## Cleaning decisions

- Text columns are stripped of leading and trailing whitespace.
- `TotalCharges` is converted with `pd.to_numeric(errors="coerce")`; blank values
  therefore become missing values rather than being silently dropped.
- `Churn` is mapped from `Yes`/`No` to `1`/`0`.
- No rows are dropped during cleaning.
- `customerID` is preserved by cleaning and may be dropped later during training.
