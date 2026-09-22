"""The awkward cases: failures, empty runs, the timing floor, threads, odd labels."""

from __future__ import annotations

import logging
import pathlib
import threading
import time

import pytest

import ml_inference_profiler as mip
from ml_inference_profiler import Profiler
from ml_inference_profiler._timing import measure_stage_overhead


# -- a stage that raises -------------------------------------------------------------
def test_a_raising_stage_still_records_its_time_and_re_raises():
    profiler = Profiler()
    with pytest.raises(ValueError, match="model exploded"):
        with profiler.stage("model"):
            time.sleep(0.01)
            raise ValueError("model exploded")

    stage = profiler.report().find("model")
    assert stage.calls == 1
    assert stage.errors == 1
    assert stage.total_ms > 0  # the elapsed time survived the exception


def test_the_original_exception_reaches_the_caller_unchanged():
    profiler = Profiler()
    original = KeyError("weights")
    try:
        with profiler.stage("load"):
            raise original
    except KeyError as caught:
        assert caught is original
    else:  # pragma: no cover - the raise above always fires
        pytest.fail("the exception was swallowed")


def test_a_failure_inside_a_pipeline_propagates_after_being_recorded():
    def boom(_):
        raise RuntimeError("bad batch")

    profiler = Profiler()
    with pytest.raises(RuntimeError, match="bad batch"):
        profiler.run([("ok", lambda x: x), ("boom", boom)], 1, repeats=1, warmup=0)

    report = profiler.report()
    assert report.find("ok").errors == 0
    assert report.find("boom").errors == 1
    assert any("raised on 1" in s for s in report.suggestions)
    assert "raised" in report.tree()


def test_a_failing_stage_still_unwinds_the_nesting():
    profiler = Profiler()
    with profiler.stage("outer"):
        with pytest.raises(ValueError):
            with profiler.stage("inner"):
                raise ValueError("x")
        with profiler.stage("after"):
            pass

    report = profiler.report()
    assert report.find("outer/after") is not None  # not nested under the failed stage
    assert report.find("outer/inner").errors == 1


# -- nothing recorded ----------------------------------------------------------------
def test_an_empty_profiler_returns_an_empty_report():
    report = Profiler("empty").report()

    assert report.stages == []
    assert report.total_ms == 0.0
    assert report.bottleneck is None
    assert report.stage_calls == 0
    assert report.overhead_total_ms == 0.0
    assert report.tree() == "empty: no stages recorded"
    assert "No stages were recorded" in report.summary()
    assert report.to_dict()["bottleneck"] is None
    assert report.to_frame().empty
    assert list(report.to_frame().columns)  # the columns are still there
    assert len(report.suggestions) == 1


def test_an_empty_pipeline_is_allowed():
    report = mip.profile_pipeline([], None, repeats=2)

    assert report.stages == []
    assert report.total_ms == 0.0
    assert report.bottleneck is None


# -- the timing floor ----------------------------------------------------------------
def test_the_per_stage_overhead_is_measured_and_reported():
    overhead = measure_stage_overhead()

    assert 0.0 < overhead < 0.001  # seconds: real, and well under a millisecond
    assert measure_stage_overhead() == overhead  # measured once, then cached

    report = mip.profile_pipeline([("x", lambda v: v)], 1, repeats=1, warmup=0)
    assert report.overhead_ms_per_stage == pytest.approx(overhead * 1000.0)
    assert report.overhead_total_ms == pytest.approx(report.overhead_ms_per_stage)
    assert "Timing floor" in report.summary()


# -- threads -------------------------------------------------------------------------
def test_threads_do_not_corrupt_each_others_nesting():
    profiler = Profiler("threaded")
    barrier = threading.Barrier(4)

    def worker(name):
        barrier.wait()
        for _ in range(5):
            with profiler.stage("shared"):
                with profiler.stage(name):
                    time.sleep(0.001)

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    report = profiler.report()
    shared = report.find("shared")
    assert shared.calls == 20  # every thread aggregated into the same stage
    assert shared.depth == 0
    for i in range(4):
        child = report.find(f"shared/t{i}")
        assert child is not None and child.calls == 5 and child.depth == 1
    assert report.threads == 4
    assert "4 threads" in report.summary()  # said out loud, not assumed away


def test_a_warmup_pass_does_not_silence_other_threads():
    """Regression: the warmup pause is thread-local, not process-wide.

    A pause stored on the profiler itself would drop every stage any other thread
    recorded while run() happened to be in a warmup pass - silently, with a report that
    still looked perfectly healthy. warmup=1 is the default, so this was the ordinary
    case for anyone profiling a server.
    """
    profiler = Profiler("server")
    in_warmup = threading.Event()
    other_done = threading.Event()

    def background():
        assert in_warmup.wait(5)
        for _ in range(10):
            with profiler.stage("background work"):
                pass
        other_done.set()

    def warm_step(value):
        in_warmup.set()
        other_done.wait(5)  # hold the warmup pass open while the other thread records
        return value

    worker = threading.Thread(target=background, daemon=True)
    worker.start()
    report = profiler.run([("warm", warm_step)], None, repeats=1, warmup=1)
    worker.join(5)

    assert other_done.is_set(), "the background thread never finished"
    background_stage = report.find("background work")
    assert background_stage is not None, "another thread's stages were dropped by warmup"
    assert background_stage.calls == 10
    # The warmup pass itself is still untimed on the thread that ran it: one timed call.
    assert report.find("warm").calls == 1
    assert report.threads == 2


def test_warmup_pause_does_not_leak_between_runs_on_one_thread():
    """The pause is restored after run(), so later hand-driven stages still record."""
    profiler = Profiler()
    profiler.run([("step", lambda x: x)], None, repeats=1, warmup=2)
    with profiler.stage("after"):
        pass

    report = profiler.report()
    assert report.find("after").calls == 1
    assert report.find("step").calls == 1  # 2 warmup passes stayed untimed


def test_thread_behaviour_is_documented():
    assert "thread" in Profiler.__doc__.lower()
    module_doc = mip.profiler.__doc__
    assert "Thread safety, honestly" in module_doc
    assert "add up to more than the run really" in module_doc  # the caveat is stated
    assert "how long each stage took" in module_doc
    assert "must be left on the same thread" in module_doc
    # The warmup caveat is part of the honesty, in the module and in the README.
    assert "Warmup passes are silenced per thread, never globally" in module_doc
    readme = (
        pathlib.Path(__file__).resolve().parents[1] / "README.md"
    ).read_text(encoding="utf-8")
    assert "warmup pass in `run()` pauses recording only on the thread" in readme


def test_self_time_versus_cumulative_time_is_documented():
    assert "self time" in mip.report.__doc__.lower()
    assert "cumulative" in mip.report.__doc__.lower()


def test_leaving_a_stage_that_was_never_entered_is_logged_not_crashed(caplog):
    profiler = Profiler("odd")
    context = profiler.stage("never entered")
    with caplog.at_level(logging.WARNING, logger="ml_inference_profiler.profiler"):
        context.__exit__(None, None, None)

    assert "never entered one" in caplog.text
    assert profiler.report().stages == []


# -- labels --------------------------------------------------------------------------
def test_non_ascii_labels_work_everywhere():
    profiler = Profiler("pipeline de démonstration")
    with profiler.stage("prétraitement"):
        with profiler.stage("токенизация"):
            pass
    report = profiler.report()

    assert report.find("prétraitement/токенизация") is not None
    assert "токенизация" in report.tree()
    assert "токенизация" in report.summary()
    assert "токенизация" in report.to_json()
    assert "токенизация" in list(report.to_frame()["label"])


def test_empty_or_non_string_labels_are_rejected():
    profiler = Profiler()
    for bad in ("", "   ", None, 7):
        with pytest.raises(ValueError, match="non-empty string"):
            profiler.stage(bad)


def test_a_stage_context_can_be_reused():
    profiler = Profiler()
    context = profiler.stage("again")
    with context:
        pass
    with context:
        pass

    assert profiler.report().find("again").calls == 2


def test_percentiles_of_a_single_call_are_that_call(fake_clock):
    profiler = Profiler()
    with profiler.stage("once"):
        pass
    stage = profiler.report().find("once")

    assert stage.p95_ms == pytest.approx(stage.total_ms)
    assert stage.median_ms == pytest.approx(stage.total_ms)
    assert stage.min_ms == stage.max_ms == stage.first_ms
