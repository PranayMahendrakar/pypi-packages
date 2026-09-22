import numpy as np
import pandas as pd
import pytest


def make_rich_df(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Mixed-dtype frame with strongly correlated age/income and a skewed score."""
    rng = np.random.default_rng(seed)
    base = rng.normal(50, 10, n)
    return pd.DataFrame(
        {
            "age": np.clip(base, 18, 90).round().astype("int64"),
            "income": (base * 1000 + rng.normal(0, 3000, n)).round(2),
            "score": rng.exponential(size=n),
            "city": rng.choice(["Pune", "Delhi", "Mumbai"], n, p=[0.5, 0.3, 0.2]),
            "active": rng.random(n) < 0.7,
            "joined": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D"),
        }
    )


@pytest.fixture
def rich_df() -> pd.DataFrame:
    return make_rich_df()
