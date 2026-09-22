"""Shared builders for the test suite. No files, no network, no randomness drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def meter(
    days: int = 21,
    seed: int = 0,
    start: str = "2026-03-01",
    tz=None,
    freq: str = "h",
    noise: float = 0.05,
) -> pd.Series:
    """A believable hourly office meter: quiet at night, busy 9 to 6."""
    index = pd.date_range(start, periods=24 * days, freq=freq, tz=tz)
    shape = 0.4 + 1.8 * ((index.hour >= 9) & (index.hour < 18))
    values = shape + np.random.default_rng(seed).normal(0.0, noise, index.size)
    return pd.Series(values, index=index, name="kwh")


@pytest.fixture
def readings() -> pd.Series:
    """Three weeks of hourly readings with one bad afternoon."""
    series = meter()
    series.iloc[300] = 9.0
    return series


@pytest.fixture
def frame(readings: pd.Series) -> pd.DataFrame:
    """The same readings as a two column table."""
    return pd.DataFrame({"recorded_at": readings.index, "kwh": readings.to_numpy()})
