"""Shared fixtures: small frames the checks can bite on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy() -> pd.DataFrame:
    """The README quickstart frame: 20 rows that trip several checks at once."""
    return pd.DataFrame(
        {
            "user_id": range(1, 21),
            "age": [23, 34, 45, None, 31, 29, 38, 52, 41, 27] * 2,
            "signup": ["2024-01-05"] * 20,
            "will_churn": [0, 1] * 10,
            "churn": [0, 1] * 10,
        }
    )


@pytest.fixture
def clean() -> pd.DataFrame:
    """A frame with nothing wrong with it."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.normal(size=200).round(3),
            "y": rng.normal(size=200).round(3),
            "city": [f"c{i % 7}" for i in range(200)],
            "label": ([0, 1] * 100),
        }
    )


@pytest.fixture
def kitchen_sink() -> pd.DataFrame:
    """A 200-row frame built so that every column-level check has something to find."""
    rng = np.random.default_rng(7)
    n = 200
    base = rng.normal(size=n).round(4)
    frame = pd.DataFrame(
        {
            "row_id": range(1, n + 1),                                  # id_like
            "x": base,                                                  # clean numeric
            "x_twin": (base * 2.0 + 0.5).round(4),                      # high_correlation with x
            # outliers + skew, spiked mid-frame so the spike does not line up with the label
            "money": np.where(np.arange(n) % 20 == 3, 5_000.0, rng.normal(10, 1, n)).round(4),
            "age": [None if i % 8 == 0 else 20 + (i % 40) for i in range(n)],            # missing
            "plan": ["free"] * (n - 1) + ["pro"],                       # near_constant
            "country": ["IN"] * n,                                      # constant
            "blank": [np.nan] * n,                                      # empty_column
            "sku": [f"sku-{i % 60}" for i in range(n)],                 # high_cardinality
            "joined": [f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)],     # dates_as_strings
            "amount_text": [str(i) for i in range(n)],                  # numeric_as_strings
            "mixed": [i if i % 2 else f"v{i}" for i in range(n)],       # mixed_types
            "label": [0] * (n - 4) + [1] * 4,                           # class_imbalance
        }
    )
    frame["leak"] = frame["label"] * 3.0 + 1.0                          # target_leakage
    return frame
