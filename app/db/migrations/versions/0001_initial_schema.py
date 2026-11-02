"""Create initial MLOps audit schema."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "prediction_events",
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_payload", postgresql.JSONB(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("predicted_probability", sa.Float(), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
    )
    op.create_index("ix_prediction_events_event_time", "prediction_events", ["event_time"])
    op.create_table(
        "outcome_labels",
        sa.Column(
            "prediction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("prediction_events.prediction_id"),
            primary_key=True,
        ),
        sa.Column("observed_label", sa.Integer(), nullable=False),
        sa.Column("label_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label_source", sa.Text(), nullable=False),
    )
    op.create_table(
        "monitoring_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("labeled_count", sa.Integer(), nullable=False),
        sa.Column("drift_share", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("drift_summary", postgresql.JSONB(), nullable=False),
        sa.Column("report_uri", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
    )
    op.create_table(
        "retraining_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trigger_source", sa.Text(), nullable=False),
        sa.Column("alert_id", sa.Text()),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("dataset_version", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("failure_reason", sa.Text()),
        sa.Column("candidate_version", sa.Text()),
        sa.Column("details", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "model_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column("metric_name", sa.Text(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("split_id", sa.Text(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("ci_low", sa.Float()),
        sa.Column("ci_high", sa.Float()),
    )
    op.create_table(
        "promotion_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_version", sa.Text(), nullable=False),
        sa.Column("champion_before", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("gate_results", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", name="uq_promotion_events_run_id"),
    )


def downgrade() -> None:
    op.drop_table("promotion_events")
    op.drop_table("model_evaluations")
    op.drop_table("retraining_runs")
    op.drop_table("monitoring_runs")
    op.drop_table("outcome_labels")
    op.drop_index("ix_prediction_events_event_time", table_name="prediction_events")
    op.drop_table("prediction_events")
