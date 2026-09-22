"""The main path: the README quickstart, and every public entry point once."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import model_benchmark
from model_benchmark import Benchmark, BenchmarkReport, ModelResult, benchmark, compare


def test_quickstart_from_the_readme():
    X = pd.DataFrame({"x": [0.1, 0.4, 0.6, 0.9] * 25})
    y = [0, 0, 1, 1] * 25
    report = benchmark(
        {
            "always_0": lambda d: [0] * len(d),
            "threshold": lambda d: (d["x"] > 0.5).astype(int),
        },
        (X, y),
        metric="accuracy",
    )
    text = report.summary()

    assert isinstance(report, BenchmarkReport)
    assert report.failures == {}
    assert report.names == ["always_0", "threshold"]
    assert report.best("score") == "threshold"
    assert report.results["threshold"].score == pytest.approx(1.0)
    assert report.results["always_0"].score == pytest.approx(0.5)
    assert "model-benchmark:" in text
    assert "accuracy" in text
    assert text.isascii()


def test_package_exports_and_version():
    assert model_benchmark.__version__ == "0.1.0"
    for name in ("benchmark", "compare", "Benchmark", "BenchmarkReport", "ModelResult", "METRICS"):
        assert hasattr(model_benchmark, name)


def test_latency_memory_and_throughput_are_measured():
    def heavy(rows):
        return [float(v) for v in range(2000)]

    report = benchmark({"heavy": heavy}, [1, 2, 3], repeats=3)
    result = report.results["heavy"]

    assert result.ok and result.error is None
    assert result.latency_mean_ms > 0
    assert result.latency_min_ms <= result.latency_p50_ms <= result.latency_max_ms
    assert result.latency_p95_ms <= result.latency_max_ms
    assert result.throughput_per_s > 0
    assert result.peak_memory_kb > 0  # a 2000 element list is visible to tracemalloc
    assert result.rss_delta_kb is None or result.rss_delta_kb >= 0
    assert result.repeats == 3 and result.warmup == 2


def test_objects_with_predict_are_supported():
    class Estimator:
        def predict(self, rows):
            return np.zeros(len(rows), dtype=int)

    report = benchmark({"estimator": Estimator()}, (np.zeros((6, 2)), np.zeros(6)), metric="accuracy")
    assert report.results["estimator"].score == pytest.approx(1.0)


def test_predict_wins_over_call_when_both_exist():
    class Both:
        def __call__(self, rows):
            raise AssertionError("__call__ should not be used when .predict exists")

        def predict(self, rows):
            return [0] * len(rows)

    report = benchmark({"both": Both()}, [1, 2, 3])
    assert report.results["both"].ok


def test_a_model_that_is_neither_callable_nor_has_predict_is_a_failure():
    report = benchmark({"nonsense": 42, "fine": lambda d: d}, [1, 2])
    assert "nonsense" in report.failures
    assert "neither callable" in report.failures["nonsense"]
    assert report.results["fine"].ok


def test_compare_convenience_returns_a_frame():
    frame = compare({"a": lambda d: d, "b": lambda d: list(d)}, [1, 2, 3], repeats=2)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["model"]) == ["a", "b"]
    assert "latency_ms" in frame.columns


def test_benchmark_class_keeps_settings():
    bench = Benchmark(metric="rmse", repeats=2, warmup=0, batch_sizes=(1, 2))
    assert bench.metric_name == "rmse"
    assert bench.higher_is_better is False
    assert "rmse" in repr(bench)

    X = np.arange(10.0).reshape(5, 2)
    y = np.arange(5.0)
    report = bench.run({"zeros": lambda a: np.zeros(len(a))}, (X, y))
    assert report.repeats == 2 and report.warmup == 0
    assert report.batch_sizes == (1, 2)
    assert report.results["zeros"].score > 0


def test_models_may_be_given_as_a_list():
    def alpha(rows):
        return rows

    def beta(rows):
        return list(rows)

    report = benchmark([alpha, beta], [1, 2, 3], repeats=2)
    assert report.names == ["alpha", "beta"]


def test_data_may_be_omitted_entirely():
    report = benchmark({"noop": lambda: 1}, repeats=2)
    assert report.results["noop"].ok
    assert report.n_samples is None
    assert "one input" in report.summary()


def test_batch_sizes_scale_the_reported_throughput():
    report = benchmark(
        {"identity": lambda a: a},
        np.arange(100.0),
        batch_sizes=(1, 10, "all"),
        repeats=3,
    )
    timings = report.results["identity"].timings
    assert sorted(timings) == [1, 10, 100]
    assert timings[10].rows_per_call == 10
    assert timings[100].rows_per_call == 100
    frame = report.batch_frame()
    assert list(frame["batch_size"]) == [1, 10, 100]
    assert report.primary_batch_size == 1


def test_batch_size_larger_than_the_data_is_clamped_and_explained():
    report = benchmark({"identity": lambda a: a}, [1, 2, 3], batch_sizes=(50,), repeats=2)
    assert report.results["identity"].timings[50].rows_per_call == 3
    assert any("larger than" in note for note in report.warnings)


def test_model_result_summary_reads_well():
    report = benchmark({"identity": lambda a: a}, [1, 2, 3], repeats=2)
    line = report.results["identity"].summary()
    assert line.startswith("identity:")
    assert "ms mean" in line and "KB peak" in line
    assert isinstance(report.results["identity"], ModelResult)
