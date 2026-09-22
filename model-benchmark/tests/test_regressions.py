"""Guards for the promises the README makes about how measurement works."""
from __future__ import annotations

import pathlib
import tracemalloc

import numpy as np
import pytest

import model_benchmark
from model_benchmark import benchmark

_SOURCE_DIR = pathlib.Path(model_benchmark.__file__).parent
_SOURCES = sorted(_SOURCE_DIR.glob("*.py"))


def test_timing_uses_perf_counter_and_never_time_time():
    core = (_SOURCE_DIR / "_core.py").read_text(encoding="utf-8")
    assert "perf_counter" in core
    for source in _SOURCES:
        text = source.read_text(encoding="utf-8")
        assert "time.time(" not in text, f"{source.name} measures with time.time"
        assert "time.clock(" not in text


def test_no_heavy_framework_is_imported_and_no_gpu_is_assumed():
    for source in _SOURCES:
        text = source.read_text(encoding="utf-8")
        for forbidden in ("import torch", "import tensorflow", "cuda", "nvidia", "gpu"):
            assert forbidden not in text.lower().replace("no gpu", ""), (
                f"{source.name} mentions {forbidden}"
            )


def test_tracemalloc_is_left_exactly_as_it_was_found():
    assert not tracemalloc.is_tracing()
    benchmark({"a": lambda d: [0] * 100, "b": lambda d: d}, [1, 2, 3], repeats=2)
    assert not tracemalloc.is_tracing()


def test_tracemalloc_already_running_is_left_running():
    tracemalloc.start()
    try:
        benchmark({"a": lambda d: [0] * 100}, [1, 2, 3], repeats=2)
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()


def test_memory_does_not_leak_between_models():
    def lean(rows):
        return 0

    def greedy(rows):
        return [0] * 200_000

    report = benchmark({"greedy": greedy, "lean": lean}, [1], repeats=1)
    # The lean model must not inherit the greedy model's allocations.
    assert report.results["greedy"].peak_memory_kb > 100
    assert report.results["lean"].peak_memory_kb < 10
    assert report.best("memory") == "lean"


def test_ordering_is_deterministic_across_runs():
    models = {"c": lambda d: d, "a": lambda d: list(d), "b": lambda d: tuple(d)}
    first = benchmark(models, [1, 2, 3], repeats=2)
    second = benchmark(models, [1, 2, 3], repeats=2)
    assert first.names == second.names == ["c", "a", "b"]
    assert list(first.to_frame()["model"]) == ["c", "a", "b"]
    assert list(first.to_dict()["results"]) == ["c", "a", "b"]


def test_ties_go_to_the_model_that_was_listed_first():
    from model_benchmark._result import BenchmarkReport, ModelResult

    report = BenchmarkReport(
        results={
            "first": ModelResult(name="first", latency_mean_ms=1.0, peak_memory_kb=2.0, score=0.5),
            "second": ModelResult(
                name="second", latency_mean_ms=1.0, peak_memory_kb=2.0, score=0.5
            ),
        },
        metric_name="accuracy",
    )
    assert report.best("latency") == "first"
    assert report.best("memory") == "first"
    assert report.best("score") == "first"


def test_every_model_receives_exactly_the_same_input():
    seen = []

    def watcher_a(rows):
        seen.append(("a", np.asarray(rows).tolist()))
        return rows

    def watcher_b(rows):
        seen.append(("b", np.asarray(rows).tolist()))
        return rows

    benchmark({"a": watcher_a, "b": watcher_b}, np.arange(6.0), warmup=0, repeats=1)
    inputs_a = [payload for name, payload in seen if name == "a"]
    inputs_b = [payload for name, payload in seen if name == "b"]
    assert inputs_a == inputs_b


def test_library_code_does_not_print(capsys):
    def broken(rows):
        raise ValueError("boom")

    benchmark({"ok": lambda d: d, "broken": broken}, [1, 2, 3], repeats=2)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_summary_stays_ascii_even_with_unicode_model_names():
    report = benchmark({"café": lambda d: d}, [1, 2], repeats=1)
    text = report.summary()
    # The names are the user's; every character the package adds is plain ASCII.
    assert text.replace("café", "cafe").isascii()
    for character in "->•→─":
        if character not in "->":
            assert character not in text


@pytest.mark.parametrize("source", _SOURCES, ids=lambda p: p.name)
def test_every_public_function_has_a_docstring(source):
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("def ") or stripped.startswith("def _"):
            continue
        body = "\n".join(lines[index : index + 12])
        assert '"""' in body, f"{source.name}:{index + 1} {stripped} has no docstring"
