"""The seven measures, each turning a recording into a number and a verdict.

Every function here takes the loaded audio and the framing already done for it
and returns one :class:`~speech_quality.report.Metric`: the raw value it found,
a 0-100 score, whether that passes, and a sentence saying what it means.

Two rules hold throughout. A measure the recording cannot support - a spectrum
of four samples, a noise floor of a signal whose level never drops - comes back
with ``measured=False`` and a neutral score rather than a made-up judgement. And
nothing here ever divides by zero: digital silence is a normal input, not an
error.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from ._frames import Frames, Spectra
from ._scoring import dbfs, piecewise
from .audio import Audio
from .report import Metric
from .thresholds import NEUTRAL_SCORE, Thresholds

__all__ = [
    "MIN_SHAPE_SAMPLES",
    "NoiseStats",
    "measure_bandwidth",
    "measure_clipping",
    "measure_dynamics",
    "measure_level",
    "measure_noise",
    "measure_silence",
    "measure_speech",
    "noise_stats",
]

MIN_SHAPE_SAMPLES = 32
"""Under this many samples a recording has a level but no shape worth measuring."""

SMOOTH_BINS = 5
"""Width of the box filter applied to a spectrum before reading its edge off."""


def _unmeasured(name: str, unit: str, reason: str) -> Metric:
    """A measure the recording could not support, scored neutrally and said so.

    ``ok`` is ``None`` rather than True: the measurement was not taken, so it
    passed nothing, and a reader scanning the pass column of a summary must not
    come away thinking it did.
    """
    return Metric(
        value=None,
        score=NEUTRAL_SCORE,
        ok=None,
        message=reason,
        name=name,
        unit=unit,
        measured=False,
        details={},
    )


def _too_short(audio: Audio, name: str, unit: str) -> Optional[Metric]:
    """The "not enough samples" answer for the measures that need a shape."""
    if audio.n_samples >= MIN_SHAPE_SAMPLES:
        return None
    return _unmeasured(
        name,
        unit,
        "the recording is {} sample(s) long, too short to measure {}".format(
            audio.n_samples, name
        ),
    )


def _rms(values: np.ndarray) -> float:
    """Root mean square of a non-empty array, as a plain float."""
    return float(math.sqrt(float(np.dot(values, values)) / float(values.size)))


def measure_level(audio: Audio, thresholds: Thresholds) -> Metric:
    """How loud the recording is, against the level a healthy one sits at.

    Args:
        audio: the loaded recording.
        thresholds: supplies the target level and the tolerance around it.

    Returns:
        A metric whose value is the RMS level in dBFS.
    """
    samples = audio.samples
    rms = _rms(samples)
    peak = float(np.max(np.abs(samples)))
    rms_db = dbfs(rms)
    peak_db = dbfs(peak)
    target = float(thresholds.target_rms_dbfs)
    tolerance = float(thresholds.level_tolerance_db)
    offset = rms_db - target
    headroom = max(0.0, -peak_db)

    score = piecewise(
        abs(offset),
        [
            (0.0, 100.0),
            (tolerance, 85.0),
            (2.0 * tolerance, 60.0),
            (3.0 * tolerance, 35.0),
            (4.0 * tolerance, 15.0),
            (6.0 * tolerance, 0.0),
        ],
    )
    ok = abs(offset) <= tolerance

    if audio.is_digital_silence:
        message = (
            "every sample is exactly zero, so there is no level at all: this is "
            "digital silence, not a quiet recording"
        )
    elif offset < -tolerance:
        message = (
            "the recording is quiet: {:.1f} dBFS RMS is {:.1f} dB under the {:.0f} dBFS "
            "target, so hiss and quantisation noise sit closer to the voice than they "
            "should; raise the gain at the source or normalise it".format(
                rms_db, -offset, target
            )
        )
    elif offset > tolerance:
        message = (
            "the recording is hot: {:.1f} dBFS RMS is {:.1f} dB over the {:.0f} dBFS "
            "target and peaks reach {:.1f} dBFS, leaving {:.1f} dB of headroom".format(
                rms_db, offset, target, peak_db, headroom
            )
        )
    else:
        message = (
            "level is healthy at {:.1f} dBFS RMS, peaks at {:.1f} dBFS, {:.1f} dB of "
            "headroom left".format(rms_db, peak_db, headroom)
        )

    return Metric(
        value=rms_db,
        score=score,
        ok=ok,
        message=message,
        name="level",
        unit="dBFS",
        details={
            "rms_dbfs": round(rms_db, 2),
            "peak_dbfs": round(peak_db, 2),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "target_dbfs": round(target, 2),
            "offset_db": round(offset, 2),
            "headroom_db": round(headroom, 2),
        },
    )


def _clipped_runs(hits: np.ndarray) -> Tuple[int, int]:
    """How many runs of consecutive full-scale samples there are, and the longest."""
    if not bool(hits.any()):
        return 0, 0
    marks = hits.astype(np.int8)
    edges = np.diff(marks, prepend=np.int8(0), append=np.int8(0))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0, 0
    lengths = ends - starts
    return int(starts.size), int(lengths.max())


def measure_clipping(audio: Audio, thresholds: Thresholds) -> Metric:
    """How much of the recording is pinned at or near full scale.

    Args:
        audio: the loaded recording.
        thresholds: supplies the full-scale level and the share still tolerated.

    Returns:
        A metric whose value is the share of clipped samples, 0.0 to 1.0.
    """
    level = float(thresholds.clip_level)
    rail = float(getattr(audio, "positive_rail", 1.0) or 1.0)
    # Each side against its own rail. With one fixed level of 0.995, an 8-bit file -
    # whose positive side tops out at 0.992 - could never register positive clipping,
    # so a recording clipped on both sides reported exactly half of it.
    hits = (audio.samples >= level * rail) | (audio.samples <= -level)
    count = int(np.count_nonzero(hits))
    share = float(count) / float(audio.n_samples)
    runs, longest = _clipped_runs(hits)
    longest_ms = 1000.0 * float(longest) / float(audio.sample_rate)
    allowed = float(thresholds.clip_ok_share)

    score = piecewise(
        share,
        [
            (0.0, 100.0),
            (allowed, 92.0),
            (0.002, 75.0),
            (0.01, 45.0),
            (0.05, 15.0),
            (0.2, 0.0),
        ],
    )
    ok = share <= allowed

    if count == 0:
        message = "no samples reach full scale, so nothing is clipped"
    elif ok:
        message = (
            "{} sample(s), {:.3%} of the recording, touch full scale in {} run(s); that "
            "is within tolerance".format(count, share, runs)
        )
    else:
        message = (
            "clipping: {:.2%} of samples are pinned at full scale across {} run(s), the "
            "longest {:.1f} ms; the waveform is flat-topped there and no processing "
            "brings it back, so re-record with more headroom".format(share, runs, longest_ms)
        )

    return Metric(
        value=share,
        score=score,
        ok=ok,
        message=message,
        name="clipping",
        unit="share",
        details={
            "clipped_samples": count,
            "clipped_share": round(share, 6),
            "runs": runs,
            "longest_run_samples": longest,
            "longest_run_ms": round(longest_ms, 3),
            "clip_level": round(float(thresholds.clip_level), 4),
        },
    )


class NoiseStats:
    """What the frame levels say about the noise floor under a recording.

    Attributes:
        floor_dbfs: level of the quiet frames, the noise floor.
        signal_dbfs: level of the loud frames, standing for the voice.
        snr_db: the distance between them.
        separable: False when the level never drops far enough for the two to
            mean different things, which is what a steady tone, a hum or
            unbroken noise looks like.
    """

    __slots__ = ("floor_dbfs", "signal_dbfs", "snr_db", "separable")

    def __init__(
        self, floor_dbfs: float, signal_dbfs: float, snr_db: float, separable: bool
    ) -> None:
        self.floor_dbfs = float(floor_dbfs)
        self.signal_dbfs = float(signal_dbfs)
        self.snr_db = float(snr_db)
        self.separable = bool(separable)

    def to_dict(self) -> dict:
        """JSON-safe mapping of the four numbers."""
        return {
            "noise_floor_dbfs": round(self.floor_dbfs, 2),
            "signal_dbfs": round(self.signal_dbfs, 2),
            "snr_db": round(self.snr_db, 2),
            "separable": bool(self.separable),
        }


def noise_stats(frames: Frames, thresholds: Thresholds) -> NoiseStats:
    """Estimate the noise floor and signal-to-noise ratio from the frame levels.

    The floor is the quietest decile of frames and the signal the loudest, which
    needs no voice-activity decision and so cannot be thrown off by one.

    Args:
        frames: the framed recording.
        thresholds: supplies the two percentiles, and the spread below which the
            two are not telling different stories.

    Returns:
        The floor, the signal level, their difference, and whether the two could
        be told apart at all.
    """
    levels = frames.dbfs
    floor = float(np.percentile(levels, float(thresholds.noise_floor_percentile)))
    signal = float(np.percentile(levels, float(thresholds.signal_percentile)))
    snr = signal - floor
    separable = bool(snr >= float(thresholds.stationary_spread_db)) and frames.count >= 4
    return NoiseStats(floor, signal, snr, separable)


def measure_noise(audio: Audio, frames: Frames, thresholds: Thresholds) -> Metric:
    """The noise floor under the voice, and how far the voice rises above it.

    Args:
        audio: the loaded recording.
        frames: the framed recording.
        thresholds: supplies the percentiles and the minimum acceptable ratio.

    Returns:
        A metric whose value is the signal-to-noise ratio in dB.
    """
    short = _too_short(audio, "noise", "dB")
    if short is not None:
        return short

    stats = noise_stats(frames, thresholds)
    details = dict(stats.to_dict())
    details.update(
        {
            "floor_percentile": float(thresholds.noise_floor_percentile),
            "signal_percentile": float(thresholds.signal_percentile),
            "frames": frames.count,
        }
    )

    if not stats.separable:
        metric = _unmeasured(
            "noise",
            "dB",
            "the level never drops: the quiet and the loud frames sit within {:.1f} dB "
            "of each other, so a noise floor cannot be told apart from the signal at "
            "all. A steady tone, mains hum or unbroken noise looks like this".format(
                stats.snr_db
            ),
        )
        metric.value = stats.snr_db
        metric.details = details
        return metric

    score = piecewise(
        stats.snr_db,
        [
            (0.0, 0.0),
            (6.0, 20.0),
            (10.0, 38.0),
            (float(thresholds.min_snr_db), 60.0),
            (20.0, 78.0),
            (30.0, 93.0),
            (45.0, 100.0),
        ],
    )
    ok = stats.snr_db >= float(thresholds.min_snr_db)
    if ok:
        message = (
            "the voice stands {:.1f} dB above a {:.1f} dBFS noise floor, clean enough to "
            "transcribe".format(stats.snr_db, stats.floor_dbfs)
        )
    else:
        message = (
            "noisy: only {:.1f} dB separates the voice from a {:.1f} dBFS noise floor, "
            "under the {:.0f} dB a transcriber wants; the background will be heard and "
            "words will be lost in it".format(
                stats.snr_db, stats.floor_dbfs, float(thresholds.min_snr_db)
            )
        )

    return Metric(
        value=stats.snr_db,
        score=score,
        ok=ok,
        message=message,
        name="noise",
        unit="dB",
        details=details,
    )


def measure_silence(audio: Audio, frames: Frames, thresholds: Thresholds) -> Metric:
    """How much of the recording is silence, and how much of it sits at the edges.

    Args:
        audio: the loaded recording.
        frames: the framed recording.
        thresholds: supplies the silence definition and the limits.

    Returns:
        A metric whose value is the total silent share, 0.0 to 1.0.
    """
    active = frames.active_mask(thresholds)
    duration = audio.duration
    rate = float(audio.sample_rate)

    if not bool(active.any()):
        leading = duration
        trailing = duration
        silent_share = 1.0
    else:
        order = np.flatnonzero(active)
        first = int(order[0])
        last = int(order[-1])
        leading = float(frames.starts[first]) / rate
        tail_start = float(frames.starts[last] + frames.length)
        trailing = max(0.0, (float(frames.n_samples) - tail_start) / rate)
        silent_share = 1.0 - float(active.mean())

    edge = max(leading, trailing)
    max_edge = float(thresholds.max_edge_silence_s)
    max_share = float(thresholds.max_silent_share)

    share_score = piecewise(
        silent_share,
        [(0.0, 100.0), (0.3, 95.0), (max_share, 70.0), (0.8, 40.0), (0.95, 10.0), (1.0, 0.0)],
    )
    edge_score = piecewise(
        edge,
        [(0.0, 100.0), (max_edge, 85.0), (3.0, 60.0), (10.0, 25.0), (30.0, 0.0)],
    )
    score = min(share_score, edge_score)
    ok = silent_share <= max_share and edge <= max_edge

    if silent_share >= 1.0:
        message = (
            "nothing in this recording rises above the silence floor: all {:.2f} s of it "
            "reads as silence".format(duration)
        )
    elif silent_share > max_share:
        message = (
            "mostly silence: {:.0%} of the recording carries nothing, with {:.2f} s at "
            "the start and {:.2f} s at the end; trim it before paying to transcribe "
            "it".format(silent_share, leading, trailing)
        )
    elif edge > max_edge:
        message = (
            "dead air at the edges: {:.2f} s before the first sound and {:.2f} s after "
            "the last, against a {:.1f} s limit; trim it".format(leading, trailing, max_edge)
        )
    else:
        message = (
            "{:.0%} silence overall, {:.2f} s at the start and {:.2f} s at the end, which "
            "is normal for speech".format(silent_share, leading, trailing)
        )

    return Metric(
        value=silent_share,
        score=score,
        ok=ok,
        message=message,
        name="silence",
        unit="share",
        details={
            "silent_share": round(silent_share, 4),
            "active_share": round(1.0 - silent_share, 4),
            "leading_s": round(leading, 3),
            "trailing_s": round(trailing, 3),
            "silent_seconds": round(silent_share * duration, 3),
            "active_frames": int(np.count_nonzero(active)),
            "frames": frames.count,
        },
    )


def _modulation_db(frames: Frames, active: np.ndarray) -> float:
    """Spread of the active frame levels in dB: speech swings, a tone does not."""
    if int(np.count_nonzero(active)) < 2:
        return 0.0
    return float(np.std(frames.dbfs[active]))


def measure_speech(
    audio: Audio, frames: Frames, spectra: Spectra, thresholds: Thresholds
) -> Metric:
    """How much of the recording actually sounds like someone talking.

    A frame counts as speech when it rises above the silence floor, its energy
    sits in the speech band, and its spectral centre of gravity is where a voice
    puts it. On top of that the level has to swing at syllable rate: a steady
    tone passes every spectral test and is still not speech.

    Args:
        audio: the loaded recording.
        frames: the framed recording.
        spectra: spectra of an evenly spread sample of those frames.
        thresholds: supplies the band, the centroid range and the limits.

    Returns:
        A metric whose value is the estimated speech share, 0.0 to 1.0.
    """
    short = _too_short(audio, "speech", "share")
    if short is not None:
        return short
    if spectra.count == 0:
        return _unmeasured(
            "speech",
            "share",
            "the recording is too short for a usable spectrum, so speech could not be "
            "told apart from any other sound in it",
        )

    active = frames.active_mask(thresholds)
    active_here = active[spectra.indices]
    centroid_ok = (spectra.centroid >= float(thresholds.speech_centroid_low_hz)) & (
        spectra.centroid <= float(thresholds.speech_centroid_high_hz)
    )
    band_ok = spectra.band_share >= float(thresholds.speech_band_share)
    speech_like = active_here & centroid_ok & band_ok
    share = float(np.mean(speech_like))
    modulation = _modulation_db(frames, active)
    steady = modulation < float(thresholds.min_modulation_db)

    voiced = spectra.total > 0.0
    mean_centroid = float(np.mean(spectra.centroid[voiced])) if bool(voiced.any()) else 0.0
    mean_band = float(np.mean(spectra.band_share[voiced])) if bool(voiced.any()) else 0.0

    score = piecewise(
        share,
        [
            (0.0, 0.0),
            (0.1, 15.0),
            (float(thresholds.min_speech_share), 55.0),
            (0.5, 75.0),
            (0.7, 90.0),
            (0.85, 100.0),
        ],
    )
    if steady:
        score = min(score, 35.0)
    ok = share >= float(thresholds.min_speech_share) and not steady

    if steady and share > 0.0:
        message = (
            "this does not behave like speech: the level swings only {:.1f} dB, holding "
            "steady where syllables would rise and fall. A tone, a hum or unbroken noise "
            "looks like this".format(modulation)
        )
    elif share <= 0.0:
        message = (
            "no part of this recording carries speech-like energy: what is here centres "
            "on {:.0f} Hz with {:.0%} of its energy in the speech band".format(
                mean_centroid, mean_band
            )
        )
    elif not ok:
        message = (
            "only {:.0%} of the recording sounds like speech, under the {:.0%} expected; "
            "the rest is background, music or handling noise".format(
                share, float(thresholds.min_speech_share)
            )
        )
    else:
        message = (
            "{:.0%} of the recording carries speech, centred on {:.0f} Hz with the level "
            "swinging {:.1f} dB between syllables".format(share, mean_centroid, modulation)
        )

    return Metric(
        value=share,
        score=score,
        ok=ok,
        message=message,
        name="speech",
        unit="share",
        details={
            "speech_share": round(share, 4),
            "modulation_db": round(modulation, 2),
            "steady_level": bool(steady),
            "mean_centroid_hz": round(mean_centroid, 1),
            "mean_band_share": round(mean_band, 4),
            "analysed_frames": spectra.count,
            "speech_frames": int(np.count_nonzero(speech_like)),
        },
    )


def measure_dynamics(audio: Audio, frames: Frames, thresholds: Thresholds) -> Metric:
    """Crest factor, and whether the life has been compressed out of the recording.

    Args:
        audio: the loaded recording.
        frames: the framed recording.
        thresholds: supplies the crest limits.

    Returns:
        A metric whose value is the crest factor in dB.
    """
    short = _too_short(audio, "dynamics", "dB")
    if short is not None:
        return short

    active = frames.active_mask(thresholds)
    if bool(active.any()):
        rms = _rms(frames.rms[active])
        peak = float(np.max(frames.peak[active]))
        scope = "over the parts that carry sound"
    else:
        rms = _rms(audio.samples)
        peak = float(np.max(np.abs(audio.samples)))
        scope = "over the whole recording"

    if rms <= 0.0 or peak <= 0.0:
        return _unmeasured("dynamics", "dB", "there is no sound to take a crest factor of")

    crest = dbfs(peak) - dbfs(rms)
    compressed = crest < float(thresholds.compressed_crest_db)
    score = piecewise(
        crest,
        [
            (0.0, 0.0),
            (3.0, 18.0),
            (float(thresholds.min_crest_db), 50.0),
            (float(thresholds.compressed_crest_db), 75.0),
            (12.0, 92.0),
            (16.0, 100.0),
            (24.0, 100.0),
            (34.0, 75.0),
            (48.0, 45.0),
        ],
    )
    ok = crest >= float(thresholds.min_crest_db)

    if not ok:
        message = (
            "squashed: a {:.1f} dB crest factor {} means the peaks barely rise above the "
            "average, which is heavy limiting, a hard-clipped source or plain noise "
            "rather than a voice".format(crest, scope)
        )
    elif compressed:
        message = (
            "heavily compressed: a {:.1f} dB crest factor {} is flatter than unprocessed "
            "speech, which usually sits above {:.0f} dB".format(
                crest, scope, float(thresholds.compressed_crest_db)
            )
        )
    elif crest > 30.0:
        message = (
            "spiky: a {:.1f} dB crest factor {} means isolated peaks tower over an "
            "otherwise quiet recording, which is what a thump or a click does".format(
                crest, scope
            )
        )
    else:
        message = (
            "a {:.1f} dB crest factor {}, the natural rise and fall of unprocessed "
            "speech".format(crest, scope)
        )

    return Metric(
        value=crest,
        score=score,
        ok=ok,
        message=message,
        name="dynamics",
        unit="dB",
        details={
            "crest_db": round(crest, 2),
            "rms_dbfs": round(dbfs(rms), 2),
            "peak_dbfs": round(dbfs(peak), 2),
            "compressed": bool(compressed),
            "measured_over_active": bool(active.any()),
        },
    )


def _smooth(power: np.ndarray, width: int) -> np.ndarray:
    """A short box filter, so one loud bin cannot decide where the band ends."""
    if power.size < width or width < 2:
        return power
    kernel = np.ones(int(width), dtype=np.float64) / float(width)
    return np.convolve(power, kernel, mode="same")


def measure_bandwidth(
    audio: Audio, frames: Frames, spectra: Spectra, thresholds: Thresholds
) -> Metric:
    """The highest frequency still carrying real energy.

    This is what catches telephone-band audio, and audio upsampled from a
    narrower original: the file says 48 kHz, the sound stops at 3.5 kHz, and
    every consonant that lives above that is already gone.

    Args:
        audio: the loaded recording.
        frames: the framed recording.
        spectra: spectra of a sample of those frames.
        thresholds: supplies the floor, the telephone band and the limits.

    Returns:
        A metric whose value is the top of the band in Hz.
    """
    short = _too_short(audio, "bandwidth", "Hz")
    if short is not None:
        return short
    if spectra.count == 0 or spectra.freqs.size < 4:
        return _unmeasured(
            "bandwidth",
            "Hz",
            "the recording is too short for a usable spectrum, so the top of its band "
            "could not be found",
        )

    active = frames.active_mask(thresholds)
    mean_power = spectra.mean_power(active[spectra.indices])
    if float(np.max(mean_power)) <= 0.0:
        return _unmeasured(
            "bandwidth",
            "Hz",
            "there is no energy at any frequency, so there is no band to measure",
        )

    smoothed = _smooth(mean_power, SMOOTH_BINS)
    peak_power = float(np.max(smoothed))
    cutoff = peak_power * (10.0 ** (-float(thresholds.bandwidth_floor_db) / 10.0))
    above = np.flatnonzero(smoothed >= cutoff)
    bandwidth = float(spectra.freqs[int(above[-1])]) if above.size else 0.0

    nyquist = float(audio.sample_rate) / 2.0
    share = bandwidth / nyquist if nyquist > 0.0 else 0.0
    telephone = bandwidth <= float(thresholds.telephone_band_hz)
    # "Upsampled" means telephone-band CONTENT sitting in a file that could have held
    # far more - a phone call saved as a 48 kHz WAV. It used to mean "fills under 55% of
    # Nyquist", which condemned every ordinary voice recorded at 48 kHz: speech lives
    # below about 8 kHz whatever the format, so a clean studio voice fills a third of
    # the band at most. The band edge that actually costs a transcriber its consonants is
    # absolute, not a share, so that is what is tested. A native 8 kHz file cannot hold
    # more than telephone band and is judged on its own merits instead.
    # The file must be able to hold meaningfully more than telephone band for the
    # narrow content to be a loss. A native 8 kHz file tops out at 4 kHz, so phone
    # audio is all it can ever carry; a 16 kHz file can reach 8 kHz, so phone audio
    # inside one really has thrown detail away.
    upsampled = bool(telephone and nyquist > 1.25 * float(thresholds.telephone_band_hz))

    score = piecewise(
        bandwidth,
        [
            (0.0, 0.0),
            (1000.0, 12.0),
            (2000.0, 32.0),
            (float(thresholds.min_bandwidth_hz), 55.0),
            (4000.0, 70.0),
            (6000.0, 86.0),
            (8000.0, 100.0),
        ],
    )
    if upsampled:
        score = min(score, 55.0)
    ok = bandwidth >= float(thresholds.min_bandwidth_hz) and not upsampled

    if upsampled:
        message = (
            "telephone-band audio in a {} Hz file: nothing above {:.0f} Hz carries energy, "
            "though the file could hold {:.0f} Hz. It was recorded or passed through a "
            "phone line and upsampled afterwards, and the consonants that lived above that "
            "edge are not coming back".format(audio.sample_rate, bandwidth, nyquist)
        )
    elif telephone and not ok:
        message = (
            "telephone-band audio: energy stops at {:.0f} Hz, so the consonants that live "
            "above it are gone and a transcriber will confuse s, f and th".format(bandwidth)
        )
    elif telephone:
        message = (
            "energy reaches {:.0f} Hz, which is telephone band and the most this {} Hz "
            "recording can hold".format(bandwidth, audio.sample_rate)
        )
    else:
        message = (
            "energy reaches {:.0f} Hz, {:.0%} of the way to the {:.0f} Hz ceiling, wide "
            "enough for clear consonants".format(bandwidth, share, nyquist)
        )

    return Metric(
        value=bandwidth,
        score=score,
        ok=ok,
        message=message,
        name="bandwidth",
        unit="Hz",
        details={
            "bandwidth_hz": round(bandwidth, 1),
            "nyquist_hz": round(nyquist, 1),
            "nyquist_share": round(share, 4),
            "telephone_band": bool(telephone),
            "upsampled": bool(upsampled),
            "floor_db": float(thresholds.bandwidth_floor_db),
        },
    )
