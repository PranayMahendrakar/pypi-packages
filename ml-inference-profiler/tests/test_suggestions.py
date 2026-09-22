"""The advice rules, driven by reports built with known numbers."""

from __future__ import annotations

import time

import pytest

from ml_inference_profiler.suggest import classify


def _joined(report):
    return " ".join(report.suggestions).lower()


def test_the_slowest_step_is_always_named_first(stage_factory, report_factory):
    report = report_factory(
        [stage_factory("fast", 10.0), stage_factory("slow", 90.0, run_total=100.0)]
    )

    assert report.suggestions[0].startswith("Slowest step: slow")
    assert "90.00 ms" in report.suggestions[0]


def test_preprocessing_dominating_is_called_out(stage_factory, report_factory):
    report = report_factory(
        [
            stage_factory("preprocess", 70.0, run_total=100.0),
            stage_factory("model", 30.0, run_total=100.0),
        ]
    )
    text = _joined(report)

    assert "preprocessing is 70% of the run" in text
    assert "the model itself accounts for 30%" in text
    assert "cache decoded inputs" in text


def test_preprocessing_is_not_blamed_when_the_model_is_slower(stage_factory, report_factory):
    report = report_factory(
        [
            stage_factory("preprocess", 40.0, run_total=100.0),
            stage_factory("model", 60.0, run_total=100.0),
        ]
    )

    assert "preprocessing is" not in _joined(report)


def test_a_p95_far_above_the_median_points_at_a_cold_cache(stage_factory, report_factory):
    report = report_factory(
        [
            stage_factory(
                "model",
                100.0,
                calls=5,
                median_ms=10.0,
                p95_ms=55.0,
                max_ms=60.0,
                first_ms=60.0,
                run_total=100.0,
            )
        ]
    )
    text = _joined(report)

    assert "p95 of 55.00 ms against a median of 10.00 ms" in text
    assert "cold cache" in text
    assert "warmup pass" in text


def test_a_late_spike_is_not_blamed_on_a_cold_cache(stage_factory, report_factory):
    report = report_factory(
        [
            stage_factory(
                "model",
                100.0,
                calls=5,
                median_ms=10.0,
                p95_ms=55.0,
                max_ms=60.0,
                first_ms=9.0,  # the first call was fast, so it is not a cold start
                run_total=100.0,
            )
        ]
    )
    text = _joined(report)

    assert "far slower than the rest" in text
    assert "cold cache" not in text
    assert "garbage collection" in text


def test_a_steady_stage_gets_no_spike_warning(stage_factory, report_factory):
    report = report_factory(
        [stage_factory("model", 100.0, calls=5, median_ms=20.0, p95_ms=21.0, run_total=100.0)]
    )

    assert "p95" not in _joined(report)


def test_per_item_work_is_told_to_batch(stage_factory, report_factory):
    report = report_factory(
        [
            stage_factory("embed one item", 60.0, calls=600, run_total=100.0),
            stage_factory("model", 40.0, run_total=100.0),
        ],
        repeats=10,
    )
    text = _joined(report)

    assert "ran 600 time(s) (60 per repeat)" in text
    assert "per-item work" in text
    assert "one batched call" in text


def test_batching_advice_does_not_invent_a_per_repeat_rate(stage_factory, report_factory):
    """Regression: with repeats == 1 the quotient IS the total.

    A profiler driven with stage()/profile() rather than run() never sets repeats, so
    "200 time(s) (200 per repeat)" presented the grand total as a per-iteration rate.
    """
    report = report_factory(
        [
            stage_factory("per item", 80.0, calls=200, run_total=100.0),
            stage_factory("model", 20.0, run_total=100.0),
        ],
        repeats=1,
    )
    text = _joined(report)

    assert "ran 200 time(s) in total" in text
    assert "per repeat" not in text
    assert "per-item work" in text


def test_batching_advice_says_per_repeat_when_run_supplied_repeats(
    stage_factory, report_factory
):
    report = report_factory(
        [stage_factory("per item", 100.0, calls=120, run_total=100.0)], repeats=4
    )

    assert "ran 120 time(s) (30 per repeat)" in _joined(report)


def test_hand_driven_batching_advice_end_to_end():
    """The same thing through the real Profiler, not a hand-built report."""
    import ml_inference_profiler as mip

    def spin(ms):
        end = time.perf_counter() + ms / 1000.0
        while time.perf_counter() < end:
            pass

    profiler = mip.Profiler("hand")
    for _ in range(3):
        with profiler.stage("preprocess"):
            for _ in range(20):
                with profiler.stage("per item"):
                    spin(0.3)
        with profiler.stage("model"):
            spin(1.0)

    report = profiler.report()
    text = " ".join(report.suggestions)

    assert report.repeats == 1  # run() was never called
    assert "per repeat" not in text
    assert "ran 60 time(s) in total" in text


def test_a_few_slow_calls_are_not_called_per_item_work(stage_factory, report_factory):
    report = report_factory([stage_factory("model", 100.0, calls=5, run_total=100.0)], repeats=5)

    assert "per-item work" not in _joined(report)


def test_visible_timing_overhead_is_admitted(stage_factory, report_factory):
    report = report_factory(
        [stage_factory("tiny", 100.0, calls=20000, run_total=100.0)],
        repeats=1,
        overhead_ms_per_stage=0.002,  # 20000 calls x 0.002 ms = 40 ms of the 100 ms
    )
    text = _joined(report)

    assert "timing overhead is about 40.000 ms of the 100.00 ms measured (40%)" in text
    assert "wrap a batch of them in a single stage" in text


def test_small_overhead_is_not_mentioned(stage_factory, report_factory):
    report = report_factory(
        [stage_factory("step", 100.0, calls=10, run_total=100.0)],
        overhead_ms_per_stage=0.001,
    )

    assert "timing overhead" not in _joined(report)


def test_stages_that_raised_are_reported(stage_factory, report_factory):
    report = report_factory([stage_factory("model", 100.0, calls=4, errors=2, run_total=100.0)])
    text = _joined(report)

    assert "raised on 2 of its 4 call(s)" in text
    assert "a run that failed" in text


def test_an_empty_report_says_how_to_start():
    from ml_inference_profiler import ProfileReport

    report = ProfileReport(name="nothing")

    assert len(report.suggestions) == 1
    assert "no stages were recorded" in report.suggestions[0].lower()
    assert "profile_pipeline()" in report.suggestions[0]


@pytest.mark.parametrize(
    "label,kind",
    [
        ("preprocess", "preprocess"),
        ("tokenize batch", "preprocess"),
        ("resize", "preprocess"),
        ("model", "model"),
        ("onnx session run", "model"),
        ("model input", "model"),  # the model check wins over "input"
        ("postprocess", "postprocess"),
        ("nms", "postprocess"),
        ("something else entirely", "other"),
        # Regression: "gpu" and "net" as bare substrings swallowed data-path stages
        # named after the device or the wire. They are whole-word hints now, checked
        # last, so the work the label describes wins over the hardware it mentions.
        ("gpu_upload", "preprocess"),
        ("network fetch", "preprocess"),
        ("h2d copy", "preprocess"),
        ("to_device transfer", "preprocess"),
        ("gpu", "model"),  # nothing else to go on: still the weak hint's job
        ("resnet50", "model"),  # trailing digits stripped, so it still reads as a net
    ],
)
def test_classify_labels(label, kind):
    assert classify(label) == kind


def test_a_device_copy_does_not_suppress_the_preprocessing_advice(
    stage_factory, report_factory
):
    """Regression: gpu_upload counted as model time, so the headline advice went quiet."""
    report = report_factory(
        [
            stage_factory("gpu_upload", 70.0, run_total=100.0),
            stage_factory("model", 20.0, run_total=100.0),
            stage_factory("postprocess", 10.0, run_total=100.0),
        ]
    )
    text = _joined(report)

    assert report.bottleneck.label == "gpu_upload"
    assert "preprocessing is 70% of the run" in text
    assert "the model itself accounts for 20%" in text


def test_suggestions_are_plain_ascii(stage_factory, report_factory):
    report = report_factory([stage_factory("preprocess", 80.0, run_total=100.0)])

    for line in report.suggestions:
        assert line.isascii()
