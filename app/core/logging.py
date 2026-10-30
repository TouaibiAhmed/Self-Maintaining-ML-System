"""JSON structured logging helpers without customer payload logging."""

from __future__ import annotations

import json
import logging as std_logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(std_logging.Formatter):
    """Format log records as JSON with a request identifier."""

    def format(self, record: std_logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        for key in ("event", "model_version", "status_code", "latency_ms"):
            value = getattr(record, key, None)
            if value is not None:
                event[key] = value
        return json.dumps(event, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    """Configure the application root logger with the JSON formatter."""
    handler = std_logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = std_logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> std_logging.Logger:
    """Return a logger configured for structured application events."""
    return std_logging.getLogger(name)


def log_event(
    logger: std_logging.Logger,
    level: int,
    message: str,
    *,
    request_id: str,
    event: str,
    **metadata: Any,
) -> None:
    """Log bounded metadata only; customer feature payloads are not accepted."""
    allowed_metadata = {
        key: value
        for key, value in metadata.items()
        if key in {"model_version", "status_code", "latency_ms"}
    }
    logger.log(
        level,
        message,
        extra={"request_id": request_id, "event": event, **allowed_metadata},
    )
