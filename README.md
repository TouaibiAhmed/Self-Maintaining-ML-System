# Self-Maintaining ML System

## Problem statement

Build an MLOps system that serves a customer-churn classifier and safely manages
the model after deployment. The system will record predictions, monitor service
health and input distributions, evaluate performance when delayed labels arrive,
and retrain and promote models only when explicit quality gates pass.

## Target definition

`Churn=1` is the positive class.

## Dataset source and license

The initial dataset is the IBM Telco Customer Churn dataset, commonly available
from public data repositories. The exact source, provenance, and license must be
confirmed before redistribution. **TODO:** document the verified source and
license.

## Success metrics

The primary model metric is F1 for the churn class, evaluated only on rows with
ground-truth labels. Candidate and champion models must be evaluated on the same
holdout sample. The current acceptance policy requires:

- candidate F1 to beat the champion by at least 0.01, or be within 0.005 while
  improving recall;
- candidate recall of at least 0.60;
- at least 200 new labeled rows, with at least 30 from each class;
- a 95% bootstrap interval for the F1 difference that is not entirely below 0;
- no monitored contract-type segment losing more than 0.05 F1.

**TODO:** document baseline target values beyond these acceptance gates.

## Architecture

The planned system separates:

- a FastAPI prediction service;
- request, prediction, label, and event persistence;
- monitoring and drift detection;
- retraining, evaluation, and champion/challenger promotion;
- traffic and delayed-label simulation;
- Prometheus, Grafana, Alertmanager, MLflow, and Docker Compose integration.

**TODO:** add the finalized component diagram and service configuration.

## How to run

The initial training command is:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\train_initial_model.py
```

Training validates the cleaned input schema, writes a 30% labeled production
pool to `data/processed/production_pool.csv`, and creates versioned artifacts
under `artifacts/`. The remaining data is split into stratified
train/validation/test sets using seed 42. The fixed reference training sample
is written to `data/reference/reference.parquet` only when it does not already
exist.

Missing numeric values are retained during cleaning and handled by median
imputation for model features. Missing categorical values are handled with
most-frequent imputation. Categorical features are one-hot encoded with
unknown-category handling. The logistic regression baseline uses
`class_weight="balanced"` to account for churn class imbalance. The validation
split selects a fixed probability threshold; the test split is evaluated once.

**TODO:** document the Docker Compose startup command, API health checks, and
full experiment workflow.

## Limitations

- The initial system is intended for local development on Windows with Docker
  Desktop, VS Code, Git, and Python 3.11.
- Synthetic traffic and injected drift demonstrate behavior but do not prove
  performance on a real production service.
- Kubernetes, GPU workloads, cloud infrastructure, canary rollout, and an
  LLM-based incident agent are outside the first release.
- Real customer PII must not be used.
