"""Shared fixtures: a temporary cache location and a clock the tests control."""

from __future__ import annotations

import pytest

from prompt_cache import cache as cache_module


class Clock:
    """A stand-in for ``time.time`` so time-to-live tests never sleep."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(cache_module, "_now", fake)
    return fake


@pytest.fixture
def cache_dir(tmp_path):
    """A folder that does not exist yet, so 'created on demand' stays testable."""
    return tmp_path / "cache-home"
