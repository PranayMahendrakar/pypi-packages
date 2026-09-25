"""The result objects: what they hold, how they read, and how they serialise."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import RATE, silence, speech_like, write_wav
from speech_quality import Metric, assess, assess_batch
from speech_quality.report import measured_coverage, overall_score
from speech_quality.thresholds import MEASURES, grade_for


def test_summary_is_plain_ascii(clean_speech):
    """No arrows, bullets or box characters, so any console can print it."""
    text = assess(clean_speech, sample_rate=RATE).summary()

    text.encode("ascii")  # raises if anything non-ASCII crept into the template
    assert "measures:" in text
    assert "issues" in text


def test_summary_carries_a_unicode_source_without_mangling_it(wav_dir):
    """The file name is data, not template, so it stays exactly as it was."""
    path = write_wav(wav_dir / "録音.wav", speech_like(seconds=0.5))

    text = assess(path).summary()

    assert "録音.wav" in text


def test_to_dict_is_json_safe_and_complete(clean_speech):
    """Everything the report knows survives a round trip through JSON."""
    report = assess(clean_speech, sample_rate=RATE)

    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))

    assert payload["grade"] == report.grade
    assert payload["usable"] is report.usable
    assert set(payload["metrics"]) == set(MEASURES)
    assert payload["metrics"]["level"]["unit"] == "dBFS"
    assert isinstance(payload["issues"], list)


def test_a_metric_that_could_not_be_taken_serialises_as_null():
    """An unmeasurable value is None in JSON, never a NaN that breaks a parser."""
    report = assess(np.array([0.25]), sample_rate=RATE)

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["metrics"]["bandwidth"]["value"] is None
    assert payload["metrics"]["bandwidth"]["measured"] is False


def test_issues_are_ordered_worst_first():
    """The first line a caller reads is the thing most wrong with the take."""
    padded = np.concatenate([silence(seconds=3.0), speech_like(seconds=1.0, rms=0.005)])

    report = assess(padded, sample_rate=RATE)

    assert len(report.issues) >= 2
    scores = [
        metric.score
        for metric in report.metrics.values()
        if not metric.ok and metric.measured
    ]
    assert report.issues[0] in [
        metric.message
        for metric in report.metrics.values()
        if metric.score == min(scores)
    ]


def test_grades_run_from_a_to_f():
    """Every band maps to the letter it is documented as."""
    assert grade_for(100.0) == "A"
    assert grade_for(85.0) == "A"
    assert grade_for(84.9) == "B"
    assert grade_for(70.0) == "B"
    assert grade_for(55.0) == "C"
    assert grade_for(40.0) == "D"
    assert grade_for(25.0) == "E"
    assert grade_for(0.0) == "F"


def test_overall_score_is_weighted_not_averaged():
    """Clipping pulls harder than dynamics, which is why it has a weight."""
    def metrics(**scores):
        return {
            name: Metric(value=0.0, score=scores.get(name, 100.0), ok=True, message="", name=name)
            for name in MEASURES
        }

    clipping_bad = overall_score(metrics(clipping=0.0))
    dynamics_bad = overall_score(metrics(dynamics=0.0))

    assert clipping_bad < dynamics_bad
    assert overall_score({}) == 0.0


def test_measured_coverage_counts_weight_not_measures():
    """Coverage is what stops placeholder scores from adding up to a verdict."""
    taken = {
        name: Metric(value=0.0, score=70.0, ok=True, message="", name=name)
        for name in MEASURES
    }
    assert measured_coverage(taken) == pytest.approx(1.0)

    for name in ("noise", "speech", "dynamics", "bandwidth"):
        taken[name].measured = False
    assert measured_coverage(taken) < 0.5
    assert measured_coverage({}) == 0.0


def test_metric_scores_are_clamped_on_the_way_in():
    """A scoring curve that overshoots cannot put 140 out of 100 in a report."""
    assert Metric(value=1.0, score=140.0, ok=True, message="x").score == 100.0
    assert Metric(value=1.0, score=-40.0, ok=False, message="x").score == 0.0


def test_metric_line_formats_each_unit_readably():
    """Shares print as percentages, big numbers lose their decimals, None prints n/a."""
    share = Metric(value=0.25, score=50.0, ok=True, message="m", name="silence", unit="share")
    hertz = Metric(value=7562.5, score=90.0, ok=True, message="m", name="bandwidth", unit="Hz")
    absent = Metric(value=None, score=70.0, ok=True, message="m", name="noise", unit="dB")

    assert "25.0%" in share.line()
    assert "7562 Hz" in hertz.line()
    assert "n/a" in absent.line()
    assert absent.line().startswith("  ok")


def test_a_measure_that_was_not_taken_does_not_read_as_a_pass():
    """"ok" in the mark column would tell a reader those measures passed. They were not taken."""
    absent = Metric(
        value=None, score=70.0, ok=None, message="nothing to work with",
        name="dynamics", unit="dB", measured=False,
    )

    assert absent.line().startswith("  --")
    assert "ok" not in absent.line().split()
    assert absent.to_dict()["ok"] is None
    assert absent.to_dict()["measured"] is False


def test_silence_does_not_show_untaken_measures_as_passing():
    """The case that surfaced it: three of the seven measures had nothing to measure."""
    report = assess(silence(seconds=2.0), sample_rate=RATE)

    text = report.summary()
    payload = report.to_dict()

    for name in ("dynamics", "bandwidth"):
        assert not report.metric(name).measured
        assert report.metric(name).ok is None
        assert payload["metrics"][name]["ok"] is None
        assert "  --   {}".format(name) in text
    assert "ok   dynamics" not in text


def test_metric_line_marks_a_failure():
    """A failing measure is visibly different at a glance in the summary."""
    failing = Metric(value=0.9, score=5.0, ok=False, message="m", name="clipping", unit="share")

    assert "FAIL" in failing.line()


def test_batch_summary_lists_the_worst_and_the_unreadable(wav_dir):
    """One text block answers "which takes do I redo" and "which did not open"."""
    good = write_wav(wav_dir / "good.wav", speech_like(seconds=1.0))

    batch = assess_batch([good, str(wav_dir / "gone.wav"), ("dead", silence(1.0))])

    text = batch.summary(worst=2)

    assert "2 recording(s)" in text
    assert "could not be read (1)" in text
    assert "dead" in text
    text.encode("ascii")


def test_batch_to_dict_is_json_safe(wav_dir):
    """A batch serialises whole, failures included."""
    batch = assess_batch([str(wav_dir / "gone.wav"), ("ok", speech_like(seconds=0.5))])

    payload = json.loads(json.dumps(batch.to_dict()))

    assert payload["count"] == 1
    assert payload["failures"][0]["source"] == "gone.wav"
    assert payload["results"][0]["source"] == "ok"


def test_batch_worst_is_stable_and_bounded():
    """Ties break by name, and asking for none returns none."""
    items = [("b", speech_like(seconds=0.5, seed=1)), ("a", speech_like(seconds=0.5, seed=1))]

    batch = assess_batch(items, sample_rate=RATE)

    assert [report.source for report in batch.worst(2)] == ["a", "b"]
    assert batch.worst(0) == []
    assert len(list(batch)) == 2


def test_report_label_falls_back_when_there_is_no_source(clean_speech):
    """An unnamed array still prints a sensible headline."""
    report = assess(clean_speech, sample_rate=RATE)

    assert report.label == "recording"
    assert report.summary().startswith("recording:")
