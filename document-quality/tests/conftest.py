"""Shared pages for the suite, generated once per session. Nothing is downloaded."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _synthetic as S  # noqa: E402

import document_quality  # noqa: E402

#: Letter-shaped page at test size: 850 x 1100, x-height 10 px, so an inked line
#: is 22 px from ascender top to descender foot.
X_HEIGHT = 10


@pytest.fixture(scope="session")
def page() -> np.ndarray:
    """A clean, straight page of word-shaped text."""
    return S.text_page(850, 1100, x_height=X_HEIGHT, margin=80)


@pytest.fixture(scope="session")
def clean_report(page):
    """The report for :func:`page` at 300 dpi."""
    return document_quality.assess(page, dpi=300)


@pytest.fixture(scope="session")
def blank() -> np.ndarray:
    """A sheet of paper with scanner grain and nothing on it."""
    return S.blank_page(850, 1100)


@pytest.fixture(scope="session")
def photo() -> np.ndarray:
    """A colour photograph."""
    return S.photograph()


def kinds(report) -> set:
    """The issue kinds on a report."""
    return {item.kind for item in report.issues}
