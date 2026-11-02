# Architecture

## Prediction event persistence

After FastAPI validates a `CustomerFeatures` request and the loaded `ModelStore`
produces a prediction, the API attempts one transaction that inserts a
`prediction_events` row. The row contains the full validated feature payload,
prediction UUID, predicted class and probability, model version from
`ModelStore`, request ID, event time, and inference latency.

The insert uses a dedicated SQLAlchemy session. It commits only after the row is
added successfully; any SQLAlchemy failure rolls the session back and closes it,
so a failed insert cannot leave partial data.

Database logging is deliberately non-blocking for prediction traffic. If the
insert fails, the API still returns the prediction with HTTP 200, logs the
failure with the request ID, and increments the in-process
`prediction_log_error_count` counter. This counter will be exported to
Prometheus in a later milestone.

`/ready` returns HTTP 503 unless both the model is loaded and `SELECT 1` succeeds
against the configured database engine. `/health` remains a process liveness
check and does not depend on either dependency.
