"""Shared, fully deterministic fixtures. No randomness, no network, no files
outside ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

#: Reference traffic starts here; the window starts three weeks later.
REFERENCE_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
WINDOW_START = datetime(2026, 9, 22, tzinfo=timezone.utc)

#: 36 seconds between records is exactly 100 records per hour.
STEP_SECONDS = 36

REFERENCE_ROWS = 200

#: The three plans the reference has ever seen.
PLANS = ("basic", "pro", "gold")


def reference_frame() -> pd.DataFrame:
    """Two hundred rows of well-behaved traffic: 90% accurate, 100/hour."""
    predictions = [index % 2 for index in range(REFERENCE_ROWS)]
    actuals = [
        1 - value if index % 10 == 0 else value
        for index, value in enumerate(predictions)
    ]
    return pd.DataFrame(
        {
            "ts": [
                REFERENCE_START + timedelta(seconds=STEP_SECONDS * index)
                for index in range(REFERENCE_ROWS)
            ],
            "prediction": predictions,
            "actual": actuals,
            "latency_ms": [10 + (index % 20) for index in range(REFERENCE_ROWS)],
            "age": [20 + (index % 40) for index in range(REFERENCE_ROWS)],
            "plan": [PLANS[index % 3] for index in range(REFERENCE_ROWS)],
        }
    )


def log_healthy(watchdog, count: int = 80) -> None:
    """Log traffic that matches the reference in every way."""
    for index in range(count):
        watchdog.log(
            features={"age": 20 + (index % 40), "plan": PLANS[index % 3]},
            prediction=index % 2,
            actual=(1 - index % 2) if index % 10 == 0 else index % 2,
            latency_ms=10 + (index % 20),
            ts=WINDOW_START + timedelta(seconds=STEP_SECONDS * index),
            request_id="req-%d" % index,
        )


@pytest.fixture
def reference() -> pd.DataFrame:
    return reference_frame()
