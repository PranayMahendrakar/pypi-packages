"""Shared helpers for the ml-feature-check test suite."""

from __future__ import annotations

from typing import List, Set

import numpy as np
import pandas as pd
import pytest

from ml_feature_check import FeatureReport


def kinds(report: FeatureReport, column: str) -> Set[str]:
    """The set of finding kinds reported for one column."""
    return {f.kind for f in report.features[column]}


def all_kinds(report: FeatureReport) -> Set[str]:
    """Every finding kind anywhere in the report."""
    return {f.kind for _, f in report.iter_findings()}


def messages(report: FeatureReport, column: str) -> List[str]:
    return [f.message for f in report.features[column]]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def quickstart_frame() -> pd.DataFrame:
    """The exact frame from the README quickstart."""
    return pd.DataFrame(
        {
            "row_id": range(60),
            "age": [23, 34, 45, 31, 29] * 12,
            "age_in_years": [23, 34, 45, 31, 29] * 12,
            "country": ["IN"] * 60,
            "signed_up": ["2024-01-05", "2024-02-11", "2024-03-18"] * 20,
            "churn_flag": [0, 1] * 30,
            "churned": [0, 1] * 30,
        }
    )
