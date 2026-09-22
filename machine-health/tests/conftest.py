"""Shared fixtures: small deterministic machines that are healthy, drifting or broken."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def machine(n_healthy=800, n_bad=200, mu=63.0, sd=1.5, seed=0):
    """A two-channel machine whose last rows run hot and noisy."""
    rng = np.random.default_rng(seed)
    temp = np.r_[rng.normal(60, 1, n_healthy), rng.normal(mu, sd, n_bad)]
    vib = np.r_[rng.normal(0.2, 0.02, n_healthy), rng.normal(0.24, 0.025, n_bad)]
    return pd.DataFrame({"temp": temp, "vibration": vib})


@pytest.fixture
def healthy_df():
    """1000 rows of a machine that never changes."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"temp": rng.normal(60, 1, 1000), "vibration": rng.normal(0.2, 0.02, 1000)}
    )


@pytest.fixture
def drifting_df():
    """1000 rows whose last fifth is hotter and noisier."""
    return machine()


@pytest.fixture
def timed_df():
    """A drifting machine with a timestamp column."""
    df = machine()
    df.insert(0, "ts", pd.date_range("2026-01-01", periods=len(df), freq="min"))
    return df


@pytest.fixture
def unicode_df():
    """Non-ASCII column names, text columns and index, to prove nothing assumes ASCII."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "温度": np.r_[rng.normal(60, 1, 80), rng.normal(70, 3, 20)],
            "vibración": rng.normal(0.2, 0.02, 100),
            "opérateur": ["Zoë", "李雷", "José", "Ñuñoa"] * 25,
        }
    )
