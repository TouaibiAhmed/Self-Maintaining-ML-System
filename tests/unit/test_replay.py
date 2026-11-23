"""Replay tests against a mocked API: no server or database needed."""

import json
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import pandas as pd
import pytest

from app.api.schemas import CustomerFeatures
from app.core.model_loader import EXPECTED_FEATURE_COLUMNS
from simulator import replay as replay_module
from simulator.pending_labels import load_pending_labels
from simulator.replay import build_traffic, replay, request_body


class FakeApi:
    """Mock /predict that records every request body and can fail on demand."""

    def __init__(self, failures: list | None = None) -> None:
        self.bodies: list[dict] = []
        self.failures = list(failures or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, json={"detail": "error"})
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "prediction_id": str(uuid4()),
                "predicted_class": 1,
                "probability": 0.9,
                "model_version": "test",
            },
        )


def _client(api: FakeApi) -> httpx.Client:
    return httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(api))


def _run(traffic: pd.DataFrame, scenario: str, api: FakeApi, path: Path, **kwargs):
    with _client(api) as client:
        return replay(
            traffic, scenario, client, pending_labels_path=path, sleep=lambda _: None, **kwargs
        )


def test_replay_never_sends_the_label(pool: pd.DataFrame, tmp_path: Path) -> None:
    api = FakeApi()
    traffic = build_traffic(pool, "baseline", 50, np.random.default_rng(1))

    summary = _run(traffic, "baseline", api, tmp_path / "labels.csv")

    assert summary.sent == 50
    assert summary.failed == 0
    for body in api.bodies:
        assert list(body) == EXPECTED_FEATURE_COLUMNS
        assert "Churn" not in body
        assert "customerID" not in body
        assert "row_id" not in body


def test_pending_labels_match_predictions_by_id(
    pool: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.csv"
    traffic = build_traffic(pool, "baseline", 20, np.random.default_rng(2))

    _run(traffic, "baseline", FakeApi(), path)

    pending = load_pending_labels(path)
    assert sorted(pending.labels.values()) == sorted(traffic["Churn"].tolist())
    rows = pd.read_csv(path)
    assert set(rows["scenario"]) == {"baseline"}
    assert rows["recorded_at"].notna().all()


def test_concept_drift_records_the_transformed_label(
    pool: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.csv"
    traffic = build_traffic(pool, "concept_drift", 400, np.random.default_rng(3))
    original = pool.set_index("row_id").loc[traffic["row_id"], "Churn"].to_numpy()
    assert (traffic["Churn"].to_numpy() != original).any()

    _run(traffic, "concept_drift", FakeApi(), path)

    recorded = pd.read_csv(path)["true_label"].to_numpy()
    np.testing.assert_array_equal(recorded, traffic["Churn"].to_numpy())


def test_missing_total_charges_is_sent_as_null(pool: pd.DataFrame) -> None:
    body = request_body(pool.iloc[0])

    assert body["TotalCharges"] is None
    assert isinstance(body["tenure"], int)
    assert isinstance(body["SeniorCitizen"], int)
    json.dumps(body, allow_nan=False)


def test_sampling_is_with_replacement_and_seeded(pool: pd.DataFrame) -> None:
    small = pool.head(10)
    first = build_traffic(small, "baseline", 100, np.random.default_rng(5))
    second = build_traffic(small, "baseline", 100, np.random.default_rng(5))

    assert len(first) == 100
    assert first["row_id"].duplicated().any()
    pd.testing.assert_frame_equal(first, second)


def test_transient_errors_are_retried_then_succeed(
    pool: pd.DataFrame,
    tmp_path: Path,
) -> None:
    api = FakeApi([503, httpx.ReadTimeout("slow")])
    traffic = build_traffic(pool, "baseline", 1, np.random.default_rng(1))

    summary = _run(traffic, "baseline", api, tmp_path / "labels.csv", max_retries=3)

    assert summary.sent == 1
    assert summary.failed == 0
    assert summary.retries == 2


def test_retries_are_bounded_and_failures_write_no_label(
    pool: pd.DataFrame,
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.csv"
    api = FakeApi([httpx.ConnectError("down")] * 10)
    traffic = build_traffic(pool, "baseline", 2, np.random.default_rng(1))

    summary = _run(traffic, "baseline", api, path, max_retries=2)

    assert summary.sent == 0
    assert summary.failed == 2
    assert summary.retries == 4
    assert not path.exists()


def test_validation_errors_are_not_retried(
    pool: pd.DataFrame,
    tmp_path: Path,
) -> None:
    api = FakeApi([422])
    traffic = build_traffic(pool, "baseline", 2, np.random.default_rng(1))

    summary = _run(traffic, "baseline", api, tmp_path / "labels.csv", max_retries=3)

    assert summary.failed == 1
    assert summary.sent == 1
    assert summary.retries == 0


def test_rate_limits_request_pace(pool: pd.DataFrame, tmp_path: Path) -> None:
    sleeps: list[float] = []
    traffic = build_traffic(pool, "baseline", 5, np.random.default_rng(1))
    with _client(FakeApi()) as client:
        replay(
            traffic,
            "baseline",
            client,
            pending_labels_path=tmp_path / "labels.csv",
            rate=10,
            sleep=sleeps.append,
            clock=lambda: 0.0,
        )

    assert sleeps == pytest.approx([0.1, 0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "argv",
    [
        ["--scenario", "baseline", "--n-rows", "0", "--seed", "1"],
        ["--scenario", "baseline", "--n-rows", "5", "--seed", "1", "--rate", "0"],
        ["--scenario", "unknown", "--n-rows", "5", "--seed", "1"],
        ["--n-rows", "5", "--seed", "1"],
    ],
)
def test_cli_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        replay_module.main(argv)


@pytest.mark.parametrize("scenario", ["baseline", "feature_drift", "concept_drift"])
def test_request_bodies_pass_api_validation(pool: pd.DataFrame, scenario: str) -> None:
    traffic = build_traffic(pool, scenario, 300, np.random.default_rng(4))

    for _, row in traffic.iterrows():
        CustomerFeatures(**request_body(row))
