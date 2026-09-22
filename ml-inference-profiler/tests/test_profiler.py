"""Recording: stages, nesting, decorators and whole pipelines."""

from __future__ import annotations

import time

import pytest

import ml_inference_profiler as mip
from ml_inference_profiler import Profiler


def test_single_stage_is_recorded(fake_clock):
    profiler = Profiler("one")
    with profiler.stage("work"):
        pass
    report = profiler.report()

    assert len(report.stages) == 1
    stage = report.stages[0]
    assert stage.label == "work"
    assert stage.path == "work"
    assert stage.calls == 1
    assert stage.depth == 0
    assert stage.parent is None
    assert stage.total_ms == pytest.approx(1.0)  # one tick of the fake clock
    assert stage.self_ms == pytest.approx(1.0)
    assert stage.share == pytest.approx(100.0)


def test_nested_stage_separates_self_time_from_cumulative(fake_clock):
    profiler = Profiler()
    with profiler.stage("outer"):
        with profiler.stage("inner"):
            pass
    report = profiler.report()

    outer = report.find("outer")
    inner = report.find("outer/inner")
    assert outer.depth == 0 and inner.depth == 1
    assert inner.parent == "outer"
    # ticks: enter outer, enter inner, leave inner, leave outer -> outer spans 3
    assert outer.total_ms == pytest.approx(3.0)
    assert inner.total_ms == pytest.approx(1.0)
    assert outer.self_ms == pytest.approx(2.0)  # cumulative minus the child
    assert inner.self_ms == pytest.approx(1.0)
    assert outer.total_ms > outer.self_ms


def test_nesting_goes_three_deep(fake_clock):
    profiler = Profiler()
    with profiler.stage("a"):
        with profiler.stage("b"):
            with profiler.stage("c"):
                pass
    report = profiler.report()

    assert [s.path for s in report.stages] == ["a", "a/b", "a/b/c"]
    assert [s.depth for s in report.stages] == [0, 1, 2]
    assert report.find("a/b/c").parent == "a/b"


def test_same_label_twice_aggregates(fake_clock):
    profiler = Profiler()
    for _ in range(3):
        with profiler.stage("model"):
            pass
    report = profiler.report()

    assert len(report.stages) == 1
    stage = report.stages[0]
    assert stage.calls == 3
    assert stage.total_ms == pytest.approx(3.0)
    assert stage.mean_ms == pytest.approx(1.0)


def test_same_label_under_different_parents_stays_separate(fake_clock):
    profiler = Profiler()
    with profiler.stage("pre"):
        with profiler.stage("copy"):
            pass
    with profiler.stage("post"):
        with profiler.stage("copy"):
            pass
    paths = [s.path for s in profiler.report().stages]

    assert paths == ["pre", "pre/copy", "post", "post/copy"]


def test_profile_decorator_uses_the_function_name():
    profiler = Profiler()

    @profiler.profile
    def run_model(x):
        """Docstring kept."""
        return x * 2

    assert run_model(3) == 6
    assert run_model(4) == 8
    assert run_model.__name__ == "run_model"
    assert run_model.__doc__ == "Docstring kept."
    assert profiler.report().find("run_model").calls == 2


def test_profile_decorator_takes_an_explicit_label():
    profiler = Profiler()

    @profiler.profile("inference")
    def anything():
        return 1

    anything()
    assert profiler.report().find("inference").calls == 1


def test_profile_decorator_nests_inside_a_stage():
    profiler = Profiler()

    @profiler.profile("child")
    def child():
        return None

    with profiler.stage("parent"):
        child()

    report = profiler.report()
    assert report.find("parent/child") is not None
    assert report.find("parent").self_ms <= report.find("parent").total_ms


def test_profile_rejects_nonsense():
    profiler = Profiler()
    with pytest.raises(TypeError):
        profiler.profile(42)
    with pytest.raises(ValueError):
        profiler.profile("   ")


def test_run_threads_data_through_the_steps():
    profiler = Profiler("chain")
    steps = [
        ("double", lambda x: [v * 2 for v in x]),
        ("total", lambda x: sum(x)),
    ]
    report = profiler.run(steps, [1, 2, 3], repeats=2, warmup=0)

    assert [s.label for s in report.stages] == ["double", "total"]
    assert all(s.calls == 2 for s in report.stages)
    assert report.repeats == 2 and report.warmup == 0


def test_warmup_runs_are_not_recorded():
    seen = []
    profiler = Profiler()
    report = profiler.run([("step", lambda x: seen.append(1))], None, repeats=2, warmup=3)

    assert len(seen) == 5  # 3 warmup + 2 timed calls actually happened
    assert report.find("step").calls == 2  # only the timed ones were recorded
    assert report.warmup == 3


def test_a_step_returning_none_passes_its_input_along():
    seen = []
    steps = [("look", lambda x: seen.append(x)), ("still", lambda x: seen.append(x))]
    mip.profile_pipeline(steps, "payload", repeats=1, warmup=0)

    assert seen == ["payload", "payload"]


def test_bare_callables_are_labelled_by_name():
    def resize(x):
        return x

    report = mip.profile_pipeline([resize, lambda x: x], 1, repeats=1, warmup=0)
    labels = [s.label for s in report.stages]

    assert labels[0] == "resize"
    assert labels[1]  # the lambda still gets a usable label


def test_run_rejects_bad_pipelines_and_counts():
    profiler = Profiler()
    with pytest.raises(ValueError, match="list of"):
        profiler.run("not a pipeline", None)
    with pytest.raises(ValueError, match="step 1"):
        profiler.run([("label", "not callable")], None)
    with pytest.raises(ValueError, match="step 1"):
        profiler.run([("", lambda x: x)], None)
    with pytest.raises(ValueError, match="repeats"):
        profiler.run([], None, repeats=0)
    with pytest.raises(ValueError, match="warmup"):
        profiler.run([], None, warmup=-1)
    with pytest.raises(ValueError, match="repeats"):
        profiler.run([], None, repeats=1.5)


def test_reset_forgets_everything(fake_clock):
    profiler = Profiler()
    with profiler.stage("gone"):
        pass
    profiler.reset()

    assert profiler.report().stages == []
    with profiler.stage("kept"):
        pass
    assert [s.label for s in profiler.report().stages] == ["kept"]


def test_report_can_be_taken_while_more_work_continues(fake_clock):
    profiler = Profiler()
    with profiler.stage("a"):
        pass
    first = profiler.report()
    with profiler.stage("a"):
        pass
    second = profiler.report()

    assert first.find("a").calls == 1
    assert second.find("a").calls == 2  # the first report is not invalidated
    assert first.find("a").calls == 1


def test_real_timing_orders_stages_by_duration():
    profiler = Profiler("real")
    with profiler.stage("slow"):
        time.sleep(0.03)
    with profiler.stage("fast"):
        time.sleep(0.001)
    report = profiler.report()

    assert report.find("slow").total_ms > report.find("fast").total_ms
    assert report.find("slow").total_ms == pytest.approx(30, abs=25)
    assert report.bottleneck.label == "slow"


def test_profile_pipeline_is_the_one_liner():
    report = mip.profile_pipeline(
        [("only", lambda x: x)], [1], name="named", repeats=3, warmup=1
    )

    assert report.name == "named"
    assert report.find("only").calls == 3
    assert isinstance(report, mip.ProfileReport)
