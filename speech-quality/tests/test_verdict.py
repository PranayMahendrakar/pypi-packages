"""The verdict: one measure failing on damage can settle it, whatever the average says.

A weighted mean cannot express "this one thing is fatal". With the shipped
weights no measure carries more than 1.5 of 7.6, so a single measure scoring
zero still leaves the overall score in the eighties: a clipped take, a
telephone-band file and a recording that is not speech at all all came back
"usable" while every per-measure flag beside them said otherwise, and the CLI
exited 0 on all three. These tests pin the verdict to the measures.

The line is between damage and a chore. Clipping, noise, a recording that does
not behave like speech and a band that stops early are none of them recoverable.
A level 6 dB off, dead air at the edges and heavy compression are, so those must
not condemn a take - a veto that fires on everything is as useless as one that
never fires.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import (
    RATE,
    clipped_voice,
    noisy_voice,
    silence,
    sine,
    speech_like,
    telephone_band,
    write_wav,
)
from speech_quality import Metric, assess
from speech_quality.cli import NOT_USABLE, USABLE, main
from speech_quality.report import build_report
from speech_quality.thresholds import DECISIVE_MEASURES, MEASURES, NEUTRAL_SCORE


def failures(report):
    """The measures that were taken and failed."""
    return sorted(
        name for name, metric in report.metrics.items() if metric.measured and not metric.ok
    )


def report_with(**outcomes):
    """A report over seven perfect measures, with the named ones changed.

    ``clipping=False`` fails that measure; ``bandwidth=None`` means it could not
    be taken at all.
    """
    metrics = {}
    for name in MEASURES:
        outcome = outcomes.get(name, True)
        if outcome is None:
            metrics[name] = Metric(
                value=None,
                score=NEUTRAL_SCORE,
                ok=None,
                message="{} had nothing to work with".format(name),
                name=name,
                measured=False,
            )
        else:
            metrics[name] = Metric(
                value=0.0,
                score=100.0 if outcome else 0.0,
                ok=outcome,
                message="{} says so".format(name),
                name=name,
            )
    return build_report(
        metrics=metrics,
        issues=[],
        notes=[],
        source="made up",
        duration=4.0,
        sample_rate=RATE,
        channels=1,
        digital_silence=False,
        usable_score=55.0,
    )


def test_a_clipped_take_that_is_good_speech_otherwise_is_not_usable():
    """The gap that let this through: clip a voice, not a sine, and nothing else breaks."""
    report = assess(clipped_voice(), sample_rate=RATE)

    assert report.metric("clipping").value > 0.002
    assert "clipping" in failures(report)
    for name in ("noise", "speech", "dynamics", "bandwidth", "silence"):
        assert report.metric(name).ok, "{} should still pass on a clipped voice".format(name)
    assert report.score > 55.0, "the weighted average alone would still have passed it"
    assert report.blocking == ["clipping"]
    assert not report.usable


def test_a_pure_tone_is_not_usable():
    """Four measures condemn it, and the average still put it over the line."""
    report = assess(sine(seconds=4.0), sample_rate=RATE)

    assert "speech" in failures(report)
    assert not report.usable
    assert "speech" in report.blocking


def test_telephone_band_audio_carried_at_48k_is_not_usable():
    """A grade-A score on a file whose consonants are already gone is a false pass."""
    report = assess(telephone_band(), sample_rate=48000)

    assert failures(report) == ["bandwidth"]
    assert report.score > 85.0, "the score is exactly why this needs a veto"
    assert report.blocking == ["bandwidth"]
    assert not report.usable


def test_a_noisy_take_is_not_usable():
    """Words lost under a hiss are lost whatever the other six measures say."""
    report = assess(noisy_voice(), sample_rate=RATE)

    assert failures(report) == ["noise"]
    assert report.metric("noise").value < 15.0
    assert report.score > 55.0
    assert report.blocking == ["noise"]
    assert not report.usable


@pytest.mark.parametrize("name", DECISIVE_MEASURES)
def test_one_decisive_measure_failing_settles_it(name):
    """Six perfect measures cannot outvote damage nothing later puts right."""
    report = report_with(**{name: False})

    assert report.score > 55.0, "the average alone would have called this usable"
    assert report.blocking == [name]
    assert not report.usable
    assert report.to_dict()["blocking"] == [name]


@pytest.mark.parametrize("name", ["level", "silence", "dynamics"])
def test_a_fault_a_later_pass_fixes_does_not_settle_it(name):
    """Gain it up, trim the ends, live with the compression: still usable."""
    report = report_with(**{name: False})

    assert report.blocking == []
    assert report.usable


def test_a_decisive_measure_that_could_not_be_taken_never_blocks():
    """Nothing to report is not the same as failing, and must not read as either."""
    report = report_with(bandwidth=None)

    assert not report.metric("bandwidth").measured
    assert report.metric("bandwidth").ok is None
    assert report.blocking == []
    assert report.usable


@pytest.mark.parametrize(
    "signal,expected",
    [
        (speech_like(seconds=4.0, rms=0.2), "level"),
        (speech_like(seconds=4.0, rms=0.02), "level"),
        (
            np.concatenate([silence(2.5), speech_like(seconds=3.0), silence(2.5)]),
            "silence",
        ),
    ],
)
def test_a_real_take_with_only_recoverable_faults_stays_usable(signal, expected):
    """Over-rejection is the other way to get this wrong, so pin that down too."""
    report = assess(signal, sample_rate=RATE)

    assert expected in failures(report)
    assert report.blocking == []
    assert report.usable


def test_a_clean_take_is_still_usable(clean_speech):
    """The everyday case the veto must not touch."""
    report = assess(clean_speech, sample_rate=RATE)

    assert report.blocking == []
    assert report.usable
    assert report.to_dict()["blocking"] == []


def test_the_summary_says_why_a_high_score_is_still_not_usable():
    """A reader seeing "grade A" and "NOT usable" is owed the reason on the next line."""
    report = assess(telephone_band(), sample_rate=48000)

    text = report.summary()

    assert "NOT usable" in text
    assert "not usable: bandwidth failed" in text
    assert "no later pass puts that right" in text
    text.encode("ascii")


def test_the_verdict_survives_json():
    """A caller filtering on the JSON gets the same answer the object gave."""
    report = assess(clipped_voice(), sample_rate=RATE)

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["usable"] is False
    assert payload["blocking"] == ["clipping"]
    assert payload["metrics"]["clipping"]["ok"] is False


def test_the_cli_redo_filter_picks_up_a_clipped_take(wav_dir, capsys):
    """The README's own idiom: speech-quality "$f" --quiet || echo "redo: $f"."""
    bad = write_wav(wav_dir / "clipped.wav", clipped_voice())
    good = write_wav(wav_dir / "clean.wav", speech_like(seconds=2.0))

    assert main([bad, "--quiet"]) == NOT_USABLE
    assert main([good, "--quiet"]) == USABLE
    assert capsys.readouterr().out == ""


def test_the_cli_reports_a_telephone_band_file_as_not_usable(wav_dir, capsys):
    """The grade stays honest about the measurement; the verdict does not follow it."""
    path = write_wav(wav_dir / "phone.wav", telephone_band(), rate=48000)

    code = main([path])

    out = capsys.readouterr().out
    assert code == NOT_USABLE
    assert "NOT usable" in out
    assert "not usable: bandwidth failed" in out
