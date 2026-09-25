"""Shared builders: small, deterministic production lines generated in memory."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytest


def line(
    days: float = 1,
    every: int = 1,
    rate: float = 60.0,
    start: str = "2026-03-02",
    hours: Optional[Tuple[int, int]] = None,
    noise: float = 0.0,
    seed: int = 0,
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """Units per ``every``-minute interval: ``rate`` while scheduled, zero outside ``hours``."""
    periods = int(round(days * 24 * 60 / every))
    t = pd.date_range(start, periods=periods, freq=f"{every}min", tz=tz)
    on = np.ones(periods, dtype=bool)
    if hours is not None:
        on = (t.hour >= hours[0]) & (t.hour < hours[1])
    units = np.full(periods, float(rate))
    if noise:
        units = units + np.random.default_rng(seed).normal(0.0, noise, periods)
    units = np.where(on, units, 0.0)
    return pd.DataFrame({"time": t, "units": units})


@pytest.fixture
def steady_day() -> pd.DataFrame:
    """One day of 1-minute counts at exactly 60 per minute, round the clock."""
    return line(days=1)
