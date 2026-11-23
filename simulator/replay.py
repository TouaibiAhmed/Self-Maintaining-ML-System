"""Replay production-pool customers into the prediction API as live traffic.

Usage (from the project root, with the API running):
    python simulator/replay.py --scenario baseline --n-rows 600 --rate 20 --seed 1

Rows are sampled with replacement from data/processed/production_pool.csv using the seed, the
scenario transformation from simulator/inject_drift.py is applied, and each row's FEATURES ONLY
are sent to POST /predict. For every successful response, (prediction_id, true_label, scenario,
timestamp) is appended to data/processed/pending_labels.csv for later label delivery. The true
label is the pool label, except in concept_drift where it is the transformed label.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow `python simulator/replay.py`: put the project root on the import path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.model_loader import EXPECTED_FEATURE_COLUMNS  # noqa: E402
from simulator.inject_drift import LABEL_COLUMN, SCENARIOS, apply_scenario  # noqa: E402
from simulator.pending_labels import (  # noqa: E402
    DEFAULT_PENDING_LABELS_PATH,
    PROJECT_ROOT,
    append_pending_label,
)

DEFAULT_POOL_PATH = PROJECT_ROOT / "data" / "processed" / "production_pool.csv"
DEFAULT_API_URL = "http://localhost:8000"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
INTEGER_FEATURES = frozenset({"SeniorCitizen", "tenure"})


@dataclass(frozen=True)
class ReplaySummary:
    """Outcome of one replay run."""

    scenario: str
    requested: int
    sent: int
    failed: int
    retries: int


class RetryableError(Exception):
    """A transient API failure that may succeed if retried."""


def load_pool(path: Path = DEFAULT_POOL_PATH) -> pd.DataFrame:
    """Load the production pool and check it has every feature and the label."""
    pool = pd.read_csv(path)
    missing = [c for c in [*EXPECTED_FEATURE_COLUMNS, LABEL_COLUMN] if c not in pool.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if pool.empty:
        raise ValueError(f"{path} has no rows")
    return pool


def build_traffic(
    pool: pd.DataFrame, scenario: str, n_rows: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample ``n_rows`` pool rows with replacement, then apply the scenario."""
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    sampled = pool.iloc[rng.integers(0, len(pool), size=n_rows)].reset_index(drop=True)
    return apply_scenario(scenario, sampled, rng)


def request_body(row: pd.Series) -> dict[str, Any]:
    """Build the /predict JSON body from model features only; the label is never included."""
    body: dict[str, Any] = {}
    for column in EXPECTED_FEATURE_COLUMNS:
        value = row[column]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            body[column] = None
        elif column in INTEGER_FEATURES:
            body[column] = int(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            body[column] = float(value)
        else:
            body[column] = str(value)
    return body


def post_with_retries(
    client: httpx.Client,
    body: dict[str, Any],
    max_retries: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any] | None, int, str | None]:
    """POST one prediction. Returns (response JSON or None, retries used, error message).

    Timeouts, connection errors and 429/5xx responses are retried up to ``max_retries`` times
    with exponential backoff. Other 4xx responses (e.g. 422 validation) are not retried.
    """
    retries = 0
    while True:
        try:
            response = client.post("/predict", json=body)
            if response.status_code in RETRYABLE_STATUS:
                raise RetryableError(f"HTTP {response.status_code}")
            if response.status_code != 200:
                return None, retries, f"HTTP {response.status_code}: {response.text[:200]}"
            payload = response.json()
            if "prediction_id" not in payload:
                return None, retries, "response has no prediction_id"
            return payload, retries, None
        except (httpx.TimeoutException, httpx.TransportError, RetryableError) as exc:
            if retries >= max_retries:
                return None, retries, f"{type(exc).__name__}: {exc}"
            sleep(backoff_seconds * (2**retries))
            retries += 1


def replay(
    traffic: pd.DataFrame,
    scenario: str,
    client: httpx.Client,
    pending_labels_path: Path = DEFAULT_PENDING_LABELS_PATH,
    rate: float = 10.0,
    max_retries: int = 3,
    backoff_seconds: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ReplaySummary:
    """Send every traffic row to the API at about ``rate`` requests per second."""
    if rate <= 0:
        raise ValueError("rate must be > 0")
    sent = failed = total_retries = 0
    started = clock()
    for index, row in enumerate(traffic.itertuples(index=False)):
        delay = started + index / rate - clock()
        if delay > 0:
            sleep(delay)
        series = pd.Series(row._asdict())
        payload, retries, error = post_with_retries(
            client, request_body(series), max_retries, backoff_seconds, sleep
        )
        total_retries += retries
        if payload is None:
            failed += 1
            print(f"row {index}: failed after {retries} retries: {error}", file=sys.stderr)
            continue
        append_pending_label(
            payload["prediction_id"],
            int(series[LABEL_COLUMN]),
            path=pending_labels_path,
            scenario=scenario,
        )
        sent += 1
    return ReplaySummary(scenario, len(traffic), sent, failed, total_retries)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--n-rows", type=int, required=True, help="Number of requests to send.")
    parser.add_argument("--rate", type=float, default=10.0, help="Requests per second.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for sampling and drift.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-request timeout (s).")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--pending-labels", type=Path, default=DEFAULT_PENDING_LABELS_PATH)
    args = parser.parse_args(argv)
    if args.n_rows < 1:
        parser.error("--n-rows must be >= 1")
    if args.rate <= 0:
        parser.error("--rate must be > 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)
    traffic = build_traffic(load_pool(args.pool), args.scenario, args.n_rows, rng)
    with httpx.Client(base_url=args.api_url, timeout=args.timeout) as client:
        summary = replay(
            traffic,
            args.scenario,
            client,
            pending_labels_path=args.pending_labels,
            rate=args.rate,
            max_retries=args.max_retries,
        )
    print(
        f"Scenario {summary.scenario}: sent {summary.sent}, failed {summary.failed} "
        f"(of {summary.requested}; {summary.retries} retries)."
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
