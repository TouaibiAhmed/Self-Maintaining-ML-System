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

## Delayed labels

### Matching contract

Predictions are matched to true outcomes by `prediction_id`, never by row order.
The label is never sent to the API. The traffic simulator (Part 4) sends a
customer to `POST /predict`, reads `prediction_id` from the response, and
appends one row to `data/processed/pending_labels.csv`:

| column | required | meaning |
| --- | --- | --- |
| `prediction_id` | yes | UUID returned by `/predict` |
| `true_label` | yes | 0 or 1 (1 = churner) |
| `scenario` | no | simulator scenario that produced the row |
| `recorded_at` | no | UTC time the row was written |

`simulator/pending_labels.py` owns this format (`append_pending_label`,
`load_pending_labels`). Rows with an invalid UUID or a label outside {0, 1} are
skipped and counted. If one `prediction_id` appears with two different labels,
it is ambiguous and is not delivered at all.

### Delivery

`simulator/deliver_labels.py --min-age-minutes N [--max M]` selects
`prediction_events` that are at least N minutes old and have no
`outcome_labels` row, keeps those that have a pending label, and inserts them
oldest first (at most M) with `label_source = "simulator"` and
`label_time = now`. The insert uses `INSERT ... ON CONFLICT (prediction_id)
DO NOTHING`, so running the script twice, or two copies at once, never creates
a duplicate and never changes an existing label. The whole run is one
transaction: if it fails, no label from that run is stored.

### Corrections

Each prediction has exactly one final label in `outcome_labels`, and delivery
never overwrites it. If a real outcome is later found to be wrong, a future
version adds a separate `label_corrections` table (prediction_id, corrected
label, reason, corrected_at, source) with one new row per correction. The
original label row stays untouched, so every past metric can still be
reproduced, and readers decide explicitly whether to apply corrections.

### Window queries and labeled metrics

`app/db/queries.py` reads a window of recent predictions
(`get_recent_window`), attaches labels with a SQL join on `prediction_id`
(`get_labeled_events`), and reports label coverage (labeled / total), median
label delay and the age of the latest label (`get_label_stats`). Coverage is
`None`, not 0, for an empty window.

`app/monitoring/labeled_metrics.compute_labeled_metrics` scores the stored
`predicted_class` against `observed_label` on labeled rows only, and always
reports `sample_count`. Below `MIN_LABELS` (default 100) the status is
`"unknown"` and precision, recall and F1 are `None`, never 0.
