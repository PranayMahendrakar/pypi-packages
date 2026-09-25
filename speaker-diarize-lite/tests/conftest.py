"""Shared recordings, generated once per test session."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _voices import ONE_VOICE, SR, TWO_VOICES, conversation  # noqa: E402


@pytest.fixture(scope="session")
def two_voices():
    """A man and a woman taking six turns; returns (signal, truth)."""
    return conversation(TWO_VOICES, seed=3)


@pytest.fixture(scope="session")
def one_voice():
    """One man talking in five stretches; returns (signal, truth)."""
    return conversation(ONE_VOICE, seed=3)


@pytest.fixture(scope="session")
def sr():
    return SR
