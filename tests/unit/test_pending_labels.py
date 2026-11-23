from pathlib import Path
from uuid import uuid4

import pytest

from simulator.pending_labels import (
    PENDING_LABEL_COLUMNS,
    append_pending_label,
    load_pending_labels,
)


def test_append_writes_header_once_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "processed" / "pending_labels.csv"
    first, second = uuid4(), uuid4()

    append_pending_label(first, 1, path=path, scenario="baseline")
    append_pending_label(str(second), 0, path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(PENDING_LABEL_COLUMNS)
    assert len(lines) == 3
    assert load_pending_labels(path).labels == {first: 1, second: 0}


@pytest.mark.parametrize("label", [2, -1, True])
def test_append_rejects_invalid_label(tmp_path: Path, label: int) -> None:
    with pytest.raises(ValueError):
        append_pending_label(uuid4(), label, path=tmp_path / "labels.csv")


def test_append_rejects_invalid_prediction_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        append_pending_label("not-a-uuid", 1, path=tmp_path / "labels.csv")


def test_missing_file_gives_no_labels(tmp_path: Path) -> None:
    assert load_pending_labels(tmp_path / "absent.csv").labels == {}


def test_hand_written_two_column_file_is_accepted(tmp_path: Path) -> None:
    prediction_id = uuid4()
    path = tmp_path / "labels.csv"
    path.write_text(f"prediction_id,true_label\n{prediction_id},1\n", encoding="utf-8")

    assert load_pending_labels(path).labels == {prediction_id: 1}


def test_invalid_and_conflicting_rows_are_skipped(tmp_path: Path) -> None:
    good, conflicting = uuid4(), uuid4()
    path = tmp_path / "labels.csv"
    path.write_text(
        "prediction_id,true_label\n"
        f"{good},0\n"
        f"{good},0\n"
        "not-a-uuid,1\n"
        f"{uuid4()},7\n"
        f"{conflicting},1\n"
        f"{conflicting},0\n",
        encoding="utf-8",
    )

    pending = load_pending_labels(path)

    assert pending.labels == {good: 0}
    assert pending.invalid_rows == 2
    assert pending.conflicting_ids == {conflicting}


def test_file_without_required_columns_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("id,label\nx,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_pending_labels(path)
