"""SQLAlchemy models for prediction, monitoring, training, and promotion records."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""


class PredictionEvent(Base):
    __tablename__ = "prediction_events"
    __table_args__ = (Index("ix_prediction_events_event_time", "event_time"),)

    prediction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    outcome: Mapped[OutcomeLabel | None] = relationship(back_populates="prediction")


class OutcomeLabel(Base):
    __tablename__ = "outcome_labels"

    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prediction_events.prediction_id"),
        primary_key=True,
    )
    observed_label: Mapped[int] = mapped_column(Integer, nullable=False)
    label_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    label_source: Mapped[str] = mapped_column(Text, nullable=False)
    prediction: Mapped[PredictionEvent] = relationship(back_populates="outcome")


class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    labeled_count: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_share: Mapped[float] = mapped_column(Float, nullable=False)
    drift_detected: Mapped[bool] = mapped_column(nullable=False)
    drift_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    report_uri: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False)


class RetrainingRun(Base):
    __tablename__ = "retraining_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    trigger_source: Mapped[str] = mapped_column(Text, nullable=False)
    alert_id: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_version: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    candidate_version: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    split_id: Mapped[str] = mapped_column(Text, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)


class PromotionEvent(Base):
    __tablename__ = "promotion_events"
    __table_args__ = (UniqueConstraint("run_id", name="uq_promotion_events_run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    candidate_version: Mapped[str] = mapped_column(Text, nullable=False)
    champion_before: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    gate_results: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
