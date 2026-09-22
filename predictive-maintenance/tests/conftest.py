"""Shared synthetic equipment histories for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_frame(
    n_rows: int = 288,
    *,
    amplitude: float = 0.7,
    ramp: int = 120,
    seed: int = 0,
    timestamped: bool = True,
) -> pd.DataFrame:
    """Sensor history that is flat then degrades over the final ``ramp`` rows."""
    wear = np.concatenate([np.zeros(n_rows - ramp), np.linspace(0.0, amplitude, ramp)])
    noise = np.random.default_rng(seed).normal(0.0, 1.0, (2, n_rows))
    data = {
        "vibration": 1 + wear + 0.15 * noise[0],
        "temp_c": 60 + 2 * wear + 0.8 * noise[1],
        "rpm": 1500.0,
    }
    if timestamped:
        data = {"time": pd.date_range("2026-01-01", periods=n_rows, freq="h"), **data}
    return pd.DataFrame(data)


def make_recovering(n_rows: int = 288, *, amplitude: float = 1.6, seed: int = 0) -> pd.DataFrame:
    """A machine that drifts away from healthy and is then brought back to it."""
    flat = n_rows // 5
    rise = (n_rows - flat) // 2
    fall = n_rows - flat - rise
    wear = np.concatenate(
        [np.zeros(flat), np.linspace(0.0, amplitude, rise), np.linspace(amplitude, 0.0, fall)]
    )
    noise = np.random.default_rng(seed).normal(0.0, 1.0, (2, n_rows))
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "vibration": 1 + wear + 0.15 * noise[0],
            "temp_c": 60 + 2 * wear + 0.8 * noise[1],
            "rpm": 1500.0,
        }
    )


def make_labelled(cycles: int = 6, life: int = 80, seed: int = 3) -> "tuple[pd.DataFrame, pd.Series]":
    """Repeated run-to-failure cycles plus the boolean failure flags."""
    rng = np.random.default_rng(seed)
    vibration, temperature, flags = [], [], []
    for _ in range(cycles):
        for step in range(life):
            wear = (step / life) ** 2
            vibration.append(1 + 3 * wear + rng.normal(0.0, 0.2))
            temperature.append(60 + 8 * wear + rng.normal(0.0, 0.6))
            flags.append(step == life - 1)
    n_rows = cycles * life
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n_rows, freq="h"),
            "vibration": vibration,
            "temp_c": temperature,
            "rpm": 1500.0,
        }
    )
    return frame, pd.Series(flags, name="failed")


@pytest.fixture
def degrading() -> pd.DataFrame:
    """A machine whose bearing is on its way out."""
    return make_frame()


@pytest.fixture
def healthy() -> pd.DataFrame:
    """A machine that is behaving exactly as it always has."""
    return make_frame(amplitude=0.0)


@pytest.fixture
def recovering() -> pd.DataFrame:
    """A machine that was drifting and has been brought back to healthy."""
    return make_recovering()


@pytest.fixture
def labelled() -> "tuple[pd.DataFrame, pd.Series]":
    """Run-to-failure history with the failures marked."""
    return make_labelled()
