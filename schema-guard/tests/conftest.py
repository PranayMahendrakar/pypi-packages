import numpy as np
import pandas as pd
import pytest

from schema_guard import Schema


@pytest.fixture
def train() -> pd.DataFrame:
    """A clean training frame covering all six dtype families."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "city": ["Oslo", "Paris", "Oslo", "Rome"],
            "score": [0.5, 0.9, 0.7, 0.1],
            "active": [True, False, True, True],
            "signup": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "tier": pd.Categorical(["a", "b", "a", "b"]),
        }
    )


@pytest.fixture
def schema(train) -> Schema:
    return Schema.infer(train)


@pytest.fixture
def messy() -> pd.DataFrame:
    """A batch that gets several things wrong relative to ``train``."""
    return pd.DataFrame(
        {
            "city": ["Paris", "Lima"],
            "id": [5.0, np.nan],
            "score": [0.3, 2.0],
            "active": ["yes", "no"],
            "signup": ["2024-05-01", "2024-06-01"],
            "tier": ["a", "c"],
            "extra": [1, 2],
        }
    )
