"""Contract for simulated ground-truth labels waiting to be delivered.

The traffic simulator (Part 4) sends a customer to POST /predict WITHOUT its label, receives a
prediction_id, and appends ``(prediction_id, true_label)`` to ``pending_labels.csv``. The label
delivery script later matches labels to stored predictions by prediction_id only, never by row
order.

File format: UTF-8 CSV with the header ``prediction_id,true_label,scenario,recorded_at``.
Only ``prediction_id`` (a UUID) and ``true_label`` (0 or 1) are required; ``scenario`` and
``recorded_at`` are optional metadata and may be empty.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PENDING_LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "pending_labels.csv"
PENDING_LABEL_COLUMNS = ("prediction_id", "true_label", "scenario", "recorded_at")
VALID_LABELS = frozenset({0, 1})


@dataclass
class PendingLabels:
    """Parsed pending labels plus the rows that could not be used."""

    labels: dict[UUID, int] = field(default_factory=dict)
    invalid_rows: int = 0
    conflicting_ids: set[UUID] = field(default_factory=set)


def append_pending_label(
    prediction_id: UUID | str,
    true_label: int,
    path: Path = DEFAULT_PENDING_LABELS_PATH,
    scenario: str | None = None,
) -> None:
    """Append one ``(prediction_id, true_label)`` pair, writing the header on first use."""
    parsed_id = UUID(str(prediction_id))
    if true_label not in VALID_LABELS or isinstance(true_label, bool):
        raise ValueError(f"true_label must be 0 or 1, got {true_label!r}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(PENDING_LABEL_COLUMNS)
        writer.writerow(
            [
                str(parsed_id),
                int(true_label),
                scenario or "",
                datetime.now(timezone.utc).isoformat(),
            ]
        )


def load_pending_labels(path: Path = DEFAULT_PENDING_LABELS_PATH) -> PendingLabels:
    """Read pending labels keyed by prediction_id.

    Rows with an invalid UUID or a label outside {0, 1} are counted and skipped. If the same
    prediction_id appears with two different labels, the label is ambiguous and that id is
    excluded entirely rather than guessing which one is right.
    """
    result = PendingLabels()
    path = Path(path)
    if not path.exists():
        return result

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"prediction_id", "true_label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                prediction_id = UUID((row.get("prediction_id") or "").strip())
                label = int((row.get("true_label") or "").strip())
            except ValueError:
                result.invalid_rows += 1
                continue
            if label not in VALID_LABELS:
                result.invalid_rows += 1
                continue
            existing = result.labels.get(prediction_id)
            if existing is not None and existing != label:
                result.conflicting_ids.add(prediction_id)
            result.labels[prediction_id] = label

    for prediction_id in result.conflicting_ids:
        result.labels.pop(prediction_id, None)
    return result
