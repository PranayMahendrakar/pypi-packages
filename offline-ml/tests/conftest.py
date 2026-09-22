"""Shared fixtures: fake machines, so the tests do not depend on the test box."""
from __future__ import annotations

import os
from typing import List, Optional

import pytest

# Never let the torch probe import a real torch during the suite.
os.environ["OFFLINE_ML_NO_TORCH"] = "1"

from offline_ml import GPU, Hardware  # noqa: E402  (after the env var above)
from offline_ml import _gpu  # noqa: E402


@pytest.fixture(autouse=True)
def _forget_probed_gpus():
    """Keep one test's fake GPU out of the next test's cache."""
    _gpu._cache = None
    yield
    _gpu._cache = None


def machine(
    *,
    ram_total_gb: float = 32.0,
    ram_available_gb: float = 24.0,
    gpus: Optional[List[GPU]] = None,
    disk_free_gb: float = 500.0,
    cpu_count: int = 8,
    platform: str = "linux",
) -> Hardware:
    """Build a Hardware record by hand."""
    return Hardware(
        cpu_count=cpu_count,
        cpu_freq_mhz=3200.0,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        gpus=list(gpus or []),
        platform=platform,
        python_version="3.10.11",
        disk_free_gb=disk_free_gb,
        disk_path="/home/tester",
    )


@pytest.fixture
def cpu_box() -> Hardware:
    """The normal case: a laptop with no GPU at all."""
    return machine()


@pytest.fixture
def cuda_box() -> Hardware:
    """A workstation with one NVIDIA card."""
    return machine(
        gpus=[GPU(name="NVIDIA GeForce RTX 4090", vram_gb=24.0, backend="cuda")],
        ram_total_gb=64.0,
        ram_available_gb=48.0,
    )


@pytest.fixture
def mps_box() -> Hardware:
    """An Apple Silicon laptop, where VRAM is the unified memory."""
    return machine(
        gpus=[GPU(name="Apple Silicon GPU (Metal)", vram_gb=16.0, backend="mps")],
        ram_total_gb=16.0,
        ram_available_gb=11.0,
        platform="macos",
    )


@pytest.fixture
def tiny_box() -> Hardware:
    """A small single-board machine that nothing in the catalogue fits on."""
    return machine(ram_total_gb=0.5, ram_available_gb=0.25, disk_free_gb=2.0, cpu_count=4)


@pytest.fixture
def models():
    """Three models the way a user would write them: plain dicts."""
    return [
        {"name": "tinyllama-1.1b-q4", "size_gb": 0.7, "quality": 3, "speed": 9},
        {"name": "mistral-7b-q4", "size_gb": 4.1, "quality": 7, "speed": 6},
        {"name": "llama-70b-q4", "size_gb": 39.0, "quality": 10, "speed": 2},
    ]
