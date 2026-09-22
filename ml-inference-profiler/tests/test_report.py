"""The report object: shares, rendering and the three export formats."""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd
import pytest

import ml_inference_profiler as mip
from ml_inference_profiler import ProfileReport, Profiler
from ml_inference_profiler.report import FRAME_COLUMNS


def _shares_by_level(report):
    levels = defaultdict(list)
    for stage in report.stages:
        levels[stage.parent].append(stage.share)
    return levels


def _nested_profiler(profiler):
    with profiler.stage("preprocess"):
        with profiler.stage("decode"):
            pass
        with profiler.stage("resize"):
            with profiler.stage("interpolate"):
                pass
    with profiler.stage("model"):
        pass
    with profiler.stage("postprocess"):
        with profiler.stage("nms"):
            pass
    return profiler.report()


def test_shares_sum_to_100_at_every_level_of_the_tree(fake_clock):
    report = _nested_profiler(Profiler("tree"))
    levels = _shares_by_level(report)

    assert len(levels) == 4  # the root level plus three parents that have children
    for parent, shares in levels.items():
        assert sum(shares) == pytest.approx(100.0, abs=1e-9), parent
        # and the rounding a reader sees on screen still adds up
        assert sum(round(s, 1) for s in shares) == pytest.approx(100.0, abs=0.2), parent


def test_shares_sum_to_100_with_real_timings():
    report = _nested_profiler(Profiler("real"))
    for parent, shares in _shares_by_level(report).items():
        assert sum(shares) == pytest.approx(100.0, abs=1e-9), parent


def test_shares_sum_to_100_even_when_every_stage_measures_zero(frozen_clock):
    profiler = Profiler("instant")
    with profiler.stage("a"):
        pass
    with profiler.stage("b"):
        pass
    report = profiler.report()

    assert report.total_ms == 0.0
    assert [s.share for s in report.stages] == [50.0, 50.0]
    assert sum(s.share for s in report.stages) == pytest.approx(100.0)
    assert all(s.pct_of_total == 0.0 for s in report.stages)  # no division by zero


def test_bottleneck_is_the_largest_self_time_not_the_largest_total(fake_clock):
    profiler = Profiler()
    with profiler.stage("wrapper"):  # big cumulative time, all of it in the child
        with profiler.stage("real work"):
            pass
    with profiler.stage("busy"):  # no children, so all of its time is self time
        pass
    report = profiler.report()

    assert report.find("wrapper").total_ms > report.find("busy").total_ms
    assert report.bottleneck.label == "wrapper"  # self 2 ticks vs busy 1 tick
    assert report.bottleneck.self_ms >= report.find("busy").self_ms


def test_pct_of_total_is_measured_against_the_whole_run(fake_clock):
    report = _nested_profiler(Profiler())
    roots = [s for s in report.stages if s.depth == 0]

    assert sum(s.pct_of_total for s in roots) == pytest.approx(100.0, abs=1e-9)
    assert sum(s.self_pct_of_total for s in report.stages) == pytest.approx(100.0, abs=1e-9)


def test_tree_is_indented_and_shows_self_time(fake_clock):
    report = _nested_profiler(Profiler("vision"))
    text = report.tree()
    lines = text.splitlines()

    assert lines[0].startswith("vision: 7 stage(s)")
    assert any(line.startswith("  preprocess") for line in lines)
    assert any(line.startswith("    decode") for line in lines)
    assert any(line.startswith("      interpolate") for line in lines)
    assert "(self " in text  # the parents report self time separately
    assert text.isascii()  # plain ASCII punctuation, no box drawing or arrows


def test_summary_explains_itself(fake_clock):
    report = _nested_profiler(Profiler("vision"))
    text = report.summary()

    assert "ml-inference-profiler: vision" in text
    assert "Bottleneck:" in text
    assert "Timing floor:" in text
    assert "Suggestions:" in text
    assert str(report) == text
    assert "vision" in repr(report) and "stage(s)" in repr(report)


def test_to_frame_has_one_row_per_stage_in_tree_order(fake_clock):
    report = _nested_profiler(Profiler())
    frame = report.to_frame()

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(FRAME_COLUMNS)
    assert len(frame) == len(report.stages)
    assert list(frame["path"]) == [s.path for s in report.stages]
    assert frame["total_ms"].ge(frame["self_ms"]).all()


def test_to_dict_is_json_safe_and_round_trips(fake_clock, tmp_path):
    report = _nested_profiler(Profiler("vision"))
    data = report.to_dict()
    text = json.dumps(data)  # would raise on a numpy scalar or a set

    assert json.loads(text)["bottleneck"] == (report.bottleneck.path)
    assert data["stage_calls"] == sum(s.calls for s in report.stages)

    path = report.save(tmp_path / "report.json")
    again = mip.load_report(path)
    assert again.name == report.name
    assert [s.path for s in again.stages] == [s.path for s in report.stages]
    assert again.total_ms == pytest.approx(report.total_ms, abs=1e-6)
    assert again.suggestions == report.suggestions


def test_saved_reports_keep_non_ascii_labels(tmp_path):
    profiler = Profiler("kärnan")
    with profiler.stage("förbehandling"):
        with profiler.stage("модель"):
            pass
    report = profiler.report()

    path = report.save(tmp_path / "unicode.json")
    raw = path.read_text(encoding="utf-8")
    assert "модель" in raw  # written with ensure_ascii=False
    assert mip.load_report(path).find("förbehandling/модель") is not None


def test_load_report_says_what_is_wrong(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="cannot read"):
        mip.load_report(missing)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        mip.load_report(broken)

    wrong = tmp_path / "wrong.json"
    wrong.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mip.load_report(wrong)


def test_find_accepts_a_label_or_a_path(fake_clock):
    report = _nested_profiler(Profiler())

    assert report.find("decode") is report.find("preprocess/decode")
    assert report.find("missing") is None


def test_report_from_dict_defaults_are_forgiving():
    report = ProfileReport.from_dict({"stages": [], "name": "x"})

    assert report.stages == []
    assert report.total_ms == 0.0
    assert report.suggestions  # an empty report still explains itself
