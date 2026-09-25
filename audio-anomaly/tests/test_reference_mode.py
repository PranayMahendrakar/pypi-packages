"""Judging against a reference: a gain offset is not a fault, the reference's own
impacts are normal, and a sustained change has to hold before it is reported."""

import re

import numpy as np
import pytest

import audio_anomaly
from audio_anomaly import _detect
from audio_anomaly import _spectra as sp
from _signals import SR, add_tone, hum

GAIN_NOTE = "than the reference on average across the spectrum"


def machine(seconds=10.0, sr=SR, seed=0):
    """A healthy machine: 120 Hz and 240 Hz hum plus a little broadband noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (
        0.3 * np.sin(2 * np.pi * 120 * t)
        + 0.15 * np.sin(2 * np.pi * 240 * t)
        + 0.01 * rng.standard_normal(t.size)
    )


def knock(x, at, amp=0.3, sr=SR, seed=0):
    """A 20 ms decaying broadband impact at ``at`` seconds."""
    y = x.copy()
    rng = np.random.default_rng(seed)
    n = int(0.02 * sr)
    i = int(at * sr)
    y[i : i + n] += amp * np.exp(-np.arange(n) / (0.004 * sr)) * rng.standard_normal(n)
    return y


def press(seconds=10.0, sr=SR, seed=0, period=0.5, amp=0.4, phase=0.25):
    """A machine whose healthy sound includes an impact every ``period`` seconds."""
    x = machine(seconds, sr, seed)
    rng = np.random.default_rng(seed + 1000)
    n = int(0.02 * sr)
    for at in np.arange(phase, seconds - 0.05, period):
        i = int(at * sr)
        x[i : i + n] += amp * np.exp(-np.arange(n) / (0.004 * sr)) * rng.standard_normal(n)
    return x


def kinds(report):
    return [e.kind for e in report.anomalies]


# --------------------------------------------------------------- gain offsets


@pytest.mark.parametrize("gain_db", [-6.0, -4.5, 4.0, 4.5, 5.0, 5.5, 6.0])
def test_a_uniform_gain_change_is_at_most_one_explained_level_change(gain_db):
    # Every band moves by the same amount, so nothing here is a new tone. The
    # 4-5 dB offsets used to flicker into a dozen 50 ms "New tone" events.
    for seed in range(2):
        x = machine(seed=20 + seed) * 10 ** (gain_db / 20)
        report = audio_anomaly.detect(x, sample_rate=SR, reference=(machine(seed=40 + seed), SR))
        assert len(report.anomalies) <= 1, report.summary()
        assert set(kinds(report)) <= {"level_rise", "level_drop"}, report.summary()
        assert any(GAIN_NOTE in n for n in report.notes), report.summary()


@pytest.mark.parametrize("gain_db", [0.0, 1.0, 2.0])
def test_a_small_gain_change_is_quiet(gain_db):
    x = machine(seed=21) * 10 ** (gain_db / 20)
    report = audio_anomaly.detect(x, sample_rate=SR, reference=(machine(seed=41), SR))
    assert report.anomalies == [], report.summary()
    assert report.max_score < report.sensitivity


@pytest.mark.parametrize("gain_db", [4.5, 5.0])
def test_a_gain_offset_does_not_hide_a_knock(gain_db):
    # Flickering tone events used to leave "edges" that swallowed real knocks:
    # 5 of 10 were missed at +4.5 dB while self mode caught them all.
    for seed in range(10):
        at = 1.0 + 0.8 * seed
        x = knock(machine(seed=20 + seed) * 10 ** (gain_db / 20), at, seed=seed)
        against_reference = audio_anomaly.detect(
            x, sample_rate=SR, reference=(machine(seed=40 + seed), SR)
        )
        against_itself = audio_anomaly.detect(x, sample_rate=SR)
        for report in (against_reference, against_itself):
            assert kinds(report) == ["burst"], report.summary()
            assert report.anomalies[0].start_s == pytest.approx(at, abs=0.01)


def test_a_new_tone_is_still_caught_on_top_of_a_gain_offset():
    x = add_tone(machine(seed=22), start=5.0, amp=0.02, freq=3100.0) * 10 ** (4.5 / 20)
    report = audio_anomaly.detect(x, sample_rate=SR, reference=(machine(seed=42), SR))
    assert kinds(report) == ["tonal"], report.summary()
    tone = report.anomalies[0]
    assert tone.start_s == pytest.approx(5.0, abs=0.1)
    assert "3.10 kHz" in tone.message


def test_a_slow_load_cycle_is_not_a_new_tone():
    # +/-30% amplitude over 5 s moves every band together.
    t = np.arange(10 * SR) / SR
    for seed in range(5):
        x = machine(seed=20 + seed) * (1.0 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
        report = audio_anomaly.detect(x, sample_rate=SR)
        assert report.anomalies == [], report.summary()


def test_a_higher_rate_reference_with_a_lower_noise_floor_is_not_a_tone():
    # Same noise per sample spread over 22 kHz instead of 8 kHz: this recording's
    # noise floor sits 4.4 dB above the reference's in every band. Not a tone.
    for seed in range(3):
        report = audio_anomaly.detect(
            machine(seed=20 + seed),
            sample_rate=SR,
            reference=(machine(sr=44100, seed=40 + seed), 44100),
        )
        assert report.anomalies == [], report.summary()
        assert any("resampled" in n for n in report.notes)


# --------------------------------------------------------------- hold and hysteresis


def test_a_sustained_change_must_hold_and_then_lasts_through_dips():
    score = np.array([0, 3.5, 3.5, 0, 0, 3.2, 3.3, 3.4, 3.1, 2.5, 3.2, 3.3, 1.0, 0])
    held = _detect._held(score, 3)
    assert held[1] < 3.0 and held[2] < 3.0  # two frames at 3.5 never hold it
    assert np.all(held[5:9] >= 3.0)
    up = _detect._hold(held >= 3.0, score >= 0.75 * 3.0)
    # held from frame 5, and carried through the dip to 2.5 until it falls to 1.0
    assert np.flatnonzero(up).tolist() == list(range(5, 12))
    both = _detect._hold(
        np.column_stack([held >= 3.0, np.zeros(score.size, dtype=bool)]),
        np.column_stack([score >= 2.25, np.ones(score.size, dtype=bool)]),
    )
    assert both[:, 0].tolist() == up.tolist()
    assert not both[:, 1].any()  # no core frame, so nothing is held


def test_held_score_matches_a_brute_force_opening():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 5, size=(60, 4))
    for length in (1, 2, 5, 12):
        fast = _detect._held(x, length)
        slow = np.zeros_like(x)
        for t in range(x.shape[0]):
            best = np.full(x.shape[1], -np.inf)
            for start in range(max(0, t - length + 1), min(t, x.shape[0] - length) + 1):
                best = np.maximum(best, x[start : start + length].min(axis=0))
            slow[t] = best
        np.testing.assert_allclose(fast, slow)


# --------------------------------------------------------------- a reference's own impacts


@pytest.mark.parametrize("period", [0.5, 0.1])
def test_impacts_the_healthy_reference_also_makes_are_not_anomalies(period):
    x = press(seed=60, period=period)
    reference = press(seed=80, period=period, phase=0.1)
    report = audio_anomaly.detect(x, sample_rate=SR, reference=(reference, SR))
    assert report.anomalies == [], report.summary()
    assert report.max_score < report.sensitivity
    assert audio_anomaly.Monitor((reference, SR)).check((x, SR)).anomalies == []


def test_without_a_reference_every_stroke_is_a_burst_with_advice():
    report = audio_anomaly.detect(press(seed=60), sample_rate=SR)
    assert len(report.of_kind("burst")) >= 15
    assert any("pass a healthy recording that contains them" in n for n in report.notes)


def test_a_knock_beyond_the_machines_own_impacts_is_still_caught():
    for seed in range(4):
        x = knock(press(seed=300 + seed), at=5.185, amp=2.0, seed=seed)
        reference = press(seed=400 + seed, phase=0.2)
        report = audio_anomaly.detect(x, sample_rate=SR, reference=(reference, SR))
        assert kinds(report) == ["burst"], report.summary()
        assert report.anomalies[0].start_s == pytest.approx(5.185, abs=0.01)
        assert "beyond the reference's own impacts" in report.anomalies[0].message
        assert any("taken as part of the machine's normal sound" in n for n in report.notes)


def test_one_stray_knock_in_the_reference_does_not_make_knocks_normal():
    reference = knock(machine(seed=41), at=3.0, amp=0.5)
    x = knock(machine(seed=21), at=6.0, amp=0.3, seed=5)
    report = audio_anomaly.detect(x, sample_rate=SR, reference=(reference, SR))
    assert kinds(report) == ["burst"], report.summary()
    assert not any("short sounds" in n for n in report.notes)


def test_a_saved_profile_cannot_carry_impacts_and_says_so():
    reference = press(seed=80, phase=0.1)
    profile = audio_anomaly.spectral_profile(reference, sample_rate=SR)
    assert profile.shape[1] == 3  # the saved form is unchanged
    report = audio_anomaly.detect(press(seed=60), sample_rate=SR, reference=profile)
    assert report.of_kind("burst")
    assert any("saved profile holds only" in n for n in report.notes)


# --------------------------------------------------------------- what the report says


def _tone_messages(report):
    pattern = r"near ([\d.]+) (k?Hz) .* the ([\d.]+) (k?Hz)-([\d.]+) (k?Hz) band"
    for event in report.of_kind("tonal"):
        match = re.search(pattern, event.message)
        if match:
            hz = [float(match.group(i)) * (1000.0 if match.group(i + 1) == "kHz" else 1.0) for i in (1, 3, 5)]
            yield event.message, hz


def test_the_named_tone_frequency_lies_inside_the_band_it_names():
    seen = 0
    for freq in (380.0, 1234.0, 5000.0):
        x = add_tone(machine(seed=23), start=5.0, amp=0.02, freq=freq)
        for reference in (None, (machine(sr=44100, seed=43), 44100)):
            report = audio_anomaly.detect(x, sample_rate=SR, reference=reference)
            for message, (f, low, high) in _tone_messages(report):
                seen += 1
                assert low - 5 <= f <= high + 5, message  # printed values are rounded
    assert seen >= 6


def test_peak_frequency_never_leaves_its_band():
    # A strong 100 Hz tone sits one bin above the lowest band (30-90 Hz at
    # 50 ms). The search used to reach a bin either side of the band.
    t = np.arange(2 * SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 100.0 * t) + 1e-4 * np.random.default_rng(0).standard_normal(t.size)
    frames = sp.analyse(x, SR, 50.0)
    every = np.arange(frames.n_frames)
    for band in range(4):
        f = _detect._peak_frequency(x, frames, every, np.zeros(0, dtype=np.int64), band)
        assert frames.layout.low_hz[band] <= f <= frames.layout.high_hz[band]


@pytest.mark.parametrize("sr,tone_hz", [(16000, 7200.0), (8000, 3800.0)])
def test_the_unmonitored_top_of_the_spectrum_is_named(sr, tone_hz):
    x = add_tone(hum(sr=sr, seconds=10.0, seed=2), sr=sr, start=5.0, amp=0.1, freq=tone_hz)
    report = audio_anomaly.detect(x, sample_rate=sr)
    note = [n for n in report.notes if "not monitored" in n]
    assert note, report.summary()
    top = float(re.search(r"reach up to ([\d.]+) kHz", note[0]).group(1)) * 1000.0
    assert top < tone_hz < sr / 2.0  # the tone it cannot see lies in the named gap
    assert "Nyquist" in note[0]


def test_no_coverage_note_when_the_rate_reaches_the_top_band():
    report = audio_anomaly.detect(hum(sr=44100, seconds=2.0, seed=2), sample_rate=44100)
    assert not any("not monitored" in n for n in report.notes)
