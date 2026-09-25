"""The detector: what "normal" is, how far each frame strays from it, and the events that come out.

Four kinds of departure are looked for, each in the way that suits it:

* **Short events** (bursts, short tonal sounds). Each frame is compared with
  the sound just before it and just after it - the louder of the two medians -
  so a knock stands out against its surroundings wherever it happens, and the
  first frames of a change that then *persists* are not mistaken for a knock.
* **Sustained changes** (a new tone, the whole sound getting louder). The band
  levels are smoothed with a running median over about 0.6 s, which a knock
  cannot move, and compared with the normal profile: the reference when one is
  given, otherwise the quieter three quarters of the recording itself. That is
  why a whine that appears halfway through is still caught without a reference:
  the typical level of its band is set by the half without it. A change has to
  *hold*: at or above the threshold for half the smoothing window before it
  counts, and it then lasts while it stays above three quarters of it, so a
  band hovering at the threshold does not flicker in and out. A tone is a band
  standing out from the rest of the spectrum, so it is judged above whatever
  shift the whole spectrum shares (a gain change moves every band together).
* **Level drops and dropouts.** The same two views on the overall level, plus
  mid-recording digital silence and runs of exact zeros as short as 5 ms.
* **Clipping**, measured on the raw samples against the format's own rail.

A reference recording also teaches the short-event view what normal is: when
the known-good machine itself makes impacts (a press, a reciprocating pump),
their strength is learned band by band, and a burst is then reported only when
it clearly exceeds them.

Every score is a robust z-score *and* has to be backed by a real change of at
least :data:`DB_PER_UNIT` dB per unit, so ``sensitivity=3.0`` means "3 sigma and
6 dB". The dB floor is what keeps a perfectly steady hum quiet: its bands barely
wobble, so their sigma is tiny, and without the floor a 0.3 dB flutter would
read as a ten-sigma event. A monitor that flags healthy machinery gets switched
off on day one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from . import _spectra as sp
from ._io import LoadedAudio, _looks_like_pair, as_profile, load_audio
from ._report import AudioAnomalyReport, Event
from ._runs import find_runs, merge_runs

logger = logging.getLogger(__name__)

#: Each unit of score has to be backed by at least this many dB of real change.
DB_PER_UNIT = 2.0

#: Smoothing window for sustained changes, seconds. A change has to last about
#: half of this to register as sustained; anything shorter is a short event.
SUSTAIN_S = 0.6

#: Frames skipped either side of the frame being judged for short events: with
#: half-overlapping frames the neighbour shares half its samples.
GAP_FRAMES = 1

#: A short event confined to one or two bands has to clear the threshold by
#: this factor, because it is the loudest of many bands and so has more chances
#: to be large by luck than a broadband average does.
NARROW_MARGIN = 1.5

#: Floor for any spread, in dB. A band that never moves at all would otherwise
#: turn a rounding error into an enormous score.
MIN_SPREAD_DB = 1.0

#: A fall this far below normal counts as the signal cutting out.
DROPOUT_DB = 40.0

#: Exact-zero runs at least this long inside the sound are dropouts.
ZERO_RUN_S = 0.005

#: A zero run only counts when the sound around it is at least this loud;
#: quiet 8-bit passages legitimately hold runs of the silence code.
ZERO_RUN_CONTEXT_DBFS = sp.SILENCE_DBFS + 20.0

#: A broadband rise shorter than this is reported as a burst, not a level rise.
BURST_MAX_S = 1.0

#: Without a reference, a band's normal level is this percentile of its
#: smoothed level: set by the quieter part, so energy that appears for up to
#: three quarters of the recording still stands out.
RISE_BASELINE_PERCENTILE = 25.0

#: Relative standard error of a MAD-based spread, times sqrt(n): the MAD has
#: about 37% efficiency, so 1 / sqrt(2 * 0.37) = 1.16.
MAD_STANDARD_ERROR = 1.16

#: The 25th minus the 5th percentile of a normal distribution, in sigmas.
LOWER_TAIL_TO_SIGMA = 1.6449 - 0.6745

#: ...and the normal overall level, for drops, is this percentile.
DROP_BASELINE_PERCENTILE = 75.0

#: When less than this share of the recording is sound, silence is the normal
#: state and silent gaps are not called dropouts.
MIN_LIVE_FRACTION = 0.5

#: A recording with fewer frames of sound than this is too short to learn a
#: typical sound from, so sustained changes are not judged without a reference.
MIN_LIVE_FRAMES_FOR_SUSTAINED = 12

#: Without a reference, less sound than this makes the learned normal rough.
ROUGH_BELOW_S = 1.5

#: When a reference profile and this recording differ by this much on average
#: across the spectrum, the report says so: it is usually gain, not the machine.
#: Half the default 6 dB floor, so any offset big enough to push bands toward
#: the threshold is explained before it is.
GAIN_NOTE_DB = 3.0

#: A sustained change, once it has held at or above the threshold for half the
#: smoothing window, lasts while its score stays above this share of it.
SUSTAIN_HYSTERESIS = 0.75

#: A reference needs at least this many short events of its own before they
#: are taken as part of the machine's normal sound.
MIN_REFERENCE_IMPACTS = 3

#: A band's spread of impact strength is never taken as less than this share
#: of the typical band's: a handful of events can agree closely by chance.
IMPACT_POOLED_FLOOR = 0.5

#: Without a reference, this many bursts or more earn a note that impacts may
#: be the machine's normal sound.
MANY_BURSTS = 6


@dataclass
class Profile:
    """The typical spectrum of a known-good recording, band by band."""

    centre_hz: np.ndarray
    level_db: np.ndarray
    spread_db: np.ndarray
    origin: str
    sample_rate: Optional[int] = None
    frame_ms: Optional[float] = None
    duration_s: Optional[float] = None
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: Per band, how far the reference's own short events (impacts, clicks)
    #: rise above their surroundings, in dB, and how much that varies from one
    #: event to the next; None when it has too few of them.
    impact_db: Optional[np.ndarray] = None
    impact_spread_db: Optional[np.ndarray] = None
    impact_count: int = 0

    def as_array(self) -> np.ndarray:
        """``(n_bands, 3)``: frequency_hz, level_db, spread_db."""
        return np.column_stack([self.centre_hz, self.level_db, self.spread_db]).astype(np.float64)

    def describe(self) -> str:
        """What this profile is, in words, for the report."""
        if self.origin == "profile":
            return "a saved spectral profile (%d bands, %.0f-%.0f Hz)" % (
                self.centre_hz.size,
                self.centre_hz[0],
                self.centre_hz[-1],
            )
        name = self.source if self.source else "a reference recording"
        return "%s (%.2f s at %d Hz)" % (name, self.duration_s or 0.0, self.sample_rate or 0)


def _check_sensitivity(sensitivity: Any) -> float:
    try:
        value = float(sensitivity)
    except (TypeError, ValueError):
        raise ValueError("sensitivity must be a number, got %r" % (sensitivity,)) from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            "sensitivity must be a positive number (3.0 is the default; higher flags "
            "less), got %r" % (sensitivity,)
        )
    return value


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 else n + 1


def _sustain_frames(hop_s: float, n_live: int) -> int:
    """Width of the sustained-change smoothing window, in frames, odd."""
    target = _odd(int(round(SUSTAIN_S / hop_s)))
    if n_live >= 4 * target:
        return target
    return max(1, min(target, _odd(n_live // 4) if n_live >= 12 else 1))


def _small_sample_factor(n: float) -> float:
    """1 plus the relative standard error of a MAD spread measured on ``n`` values."""
    return 1.0 + MAD_STANDARD_ERROR / np.sqrt(max(1.0, float(n)))


def _score(change_db: np.ndarray, spread_db: np.ndarray) -> np.ndarray:
    """Robust z of a change, capped by its size in dB; never negative."""
    z = change_db / spread_db
    return np.clip(np.minimum(z, change_db / DB_PER_UNIT), 0.0, None)


def _run_extreme(x: np.ndarray, length: int, fn: Any) -> np.ndarray:
    """``fn`` (np.minimum or np.maximum) over ``x[t : t + length]`` along axis 0.

    Built by doubling, so memory stays at the size of ``x`` whatever the length.
    Returns ``x.shape[0] - length + 1`` rows.
    """
    n = x.shape[0]
    p = 1
    cur = x
    while 2 * p <= length:
        cur = fn(cur[:-p], cur[p:])
        p *= 2
    count = n - length + 1
    return fn(cur[:count], cur[length - p : length - p + count])


def _held(score: np.ndarray, length: int) -> np.ndarray:
    """The score each frame holds for ``length`` consecutive frames, along axis 0.

    A morphological opening: for every frame, the best minimum over the
    ``length``-frame windows that contain it. A score that only touches a value
    for a frame or two does not hold it.
    """
    n = score.shape[0]
    if length <= 1 or n == 0:
        return np.array(score, dtype=np.float64, copy=True)
    if n < length:
        return np.zeros(score.shape, dtype=np.float64)
    lows = _run_extreme(np.asarray(score, dtype=np.float64), length, np.minimum)
    pad = np.full((length - 1,) + lows.shape[1:], -np.inf)
    return _run_extreme(np.concatenate([pad, lows, pad]), length, np.maximum)


def _hold(core: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Hysteresis along axis 0, column by column: the runs of ``low`` that hold a ``core`` frame."""
    shape = core.shape
    n = shape[0]
    core2 = core.reshape(n, -1)
    low2 = (low | core).reshape(n, -1)
    # One False row after every column keeps runs from joining across columns.
    sep = np.zeros((1, core2.shape[1]), dtype=bool)
    lf = np.concatenate([low2, sep]).ravel(order="F")
    cf = np.concatenate([core2, sep]).ravel(order="F")
    edges = np.diff(np.concatenate(([0], lf.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    csum = np.concatenate(([0], np.cumsum(cf, dtype=np.int64)))
    keep = (csum[stops] - csum[starts]) > 0
    delta = np.zeros(lf.size + 1, dtype=np.int64)
    np.add.at(delta, starts[keep], 1)
    np.add.at(delta, stops[keep], -1)
    marked = np.cumsum(delta[:-1]) > 0
    return marked.reshape(core2.shape[1], n + 1).T[:n].reshape(shape)


def _fast_rise(frames: sp.Frames) -> Tuple[np.ndarray, np.ndarray]:
    """``(rise_db, sigma_db)`` for short events: per frame and band, and per band.

    Each frame is compared with the louder of the medians just before and just
    after it; the spread is measured in ways a knock cannot inflate.
    """
    band = frames.band_db
    live = frames.live
    n = frames.n_frames
    nb = frames.n_bands
    n_live = int(live.sum())
    mostly_live = n_live >= MIN_LIVE_FRACTION * n
    width = _sustain_frames(frames.hop_seconds, n_live)
    side = 2 * _sustain_frames(frames.hop_seconds, n)
    # A spread measured on few frames is itself uncertain; widen it by one
    # standard error of the MAD so short recordings are not trigger-happy.
    widen = _small_sample_factor(n_live if mostly_live else n)
    if mostly_live:
        sigma = widen * np.maximum(
            np.maximum(sp.residual_spread(band[live], width), sp.lag_spread(band[live], 2)),
            MIN_SPREAD_DB,
        )
    else:
        # Silence is the normal state here and it does not wobble at all, so
        # every sound is judged against the floor spread.
        sigma = np.full(nb, widen * MIN_SPREAD_DB)
    before = sp.window_median(band, -(GAP_FRAMES + side), side)
    after = sp.window_median(band, GAP_FRAMES + 1, side)
    return band - np.maximum(before, after), sigma


def _short_event_scores(
    rise: np.ndarray, sigma: np.ndarray, live: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-band scores, the broadband burst score and the narrowband score of every frame."""
    nb = rise.shape[1]
    k_broad = min(nb, max(3, nb // 4))
    s = _score(rise, sigma)
    s[~live] = 0.0
    burst = np.sort(s, axis=1)[:, -k_broad:].mean(axis=1)
    narrow = s.max(axis=1) / NARROW_MARGIN
    return s, burst, narrow


def impact_envelope(
    frames: sp.Frames, sensitivity: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """How strongly a known-good recording's own short events rise, band by band.

    Returns ``(rise_db, spread_db, count)``: per band, the typical (median)
    rise of the events above their surroundings at each event's peak frame,
    how much that varies from one event to the next (widened for few events),
    and how many events there were. The arrays are None when there are fewer
    than :data:`MIN_REFERENCE_IMPACTS`: those are taken as incidents in the
    reference rather than as the machine's normal sound. A short sound is
    later judged by how far it exceeds the typical event in these spreads,
    averaged over its strongest bands, exactly as a burst is judged without
    a reference.
    """
    if frames.n_frames == 0 or not frames.live.any():
        return None, None, 0
    rise, sigma = _fast_rise(frames)
    _, burst, narrow = _short_event_scores(rise, sigma, frames.live)
    runs = merge_runs(find_runs((burst >= sensitivity) | (narrow >= sensitivity)), GAP_FRAMES)
    if len(runs) < MIN_REFERENCE_IMPACTS:
        return None, None, len(runs)
    peaks = np.array([rise[a + int(np.argmax(rise[a:b].mean(axis=1)))] for a, b in runs])
    level = np.maximum(np.median(peaks, axis=0), 0.0)
    spread = sp.robust_spread(peaks)
    pooled = IMPACT_POOLED_FLOOR * float(np.median(spread))
    spread = _small_sample_factor(len(runs)) * np.maximum(np.maximum(spread, pooled), MIN_SPREAD_DB)
    return level, spread, len(runs)


def _by_centre(
    ref_c: np.ndarray, values: np.ndarray, test_c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """``values`` of the reference bands whose centres match ``test_c`` (0 elsewhere), and which matched."""
    idx = np.clip(np.searchsorted(ref_c, test_c), 0, ref_c.size - 1)
    left = np.clip(idx - 1, 0, ref_c.size - 1)
    closer_left = np.abs(np.log(ref_c[left] / test_c)) < np.abs(np.log(ref_c[idx] / test_c))
    best = np.where(closer_left, left, idx)
    exact = np.abs(np.log(ref_c[best] / test_c)) < 0.01
    return np.where(exact, values[best], 0.0), exact


# --------------------------------------------------------------------------- profile


def profile_from_frames(frames: sp.Frames) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(level_db, spread_db)`` of the smoothed band levels, or None for silence."""
    live = frames.live
    n_live = int(live.sum())
    if n_live == 0:
        return None
    bands = frames.band_db[live]
    width = _sustain_frames(frames.hop_seconds, n_live)
    smooth = sp.window_median(bands, -(width // 2), width)
    level = np.median(smooth, axis=0)
    spread = sp.robust_spread(smooth)
    return level, spread


def build_profile(
    audio: Any,
    sample_rate: Optional[float],
    frame_ms: float,
    label: str,
    sensitivity: float = 3.0,
) -> Profile:
    """Load and profile a known-good recording, or accept a saved profile array.

    A recording also has its own short events measured (see
    :func:`impact_envelope`), found at ``sensitivity``.

    Raises:
        ValueError: the recording is shorter than one frame or silent throughout,
            so there is no typical spectrum to learn.
    """
    table = as_profile(audio)
    if table is not None:
        return Profile(
            centre_hz=table[:, 0].copy(),
            level_db=table[:, 1].copy(),
            spread_db=table[:, 2].copy(),
            origin="profile",
        )
    loaded = load_audio(audio, sample_rate, label=label)
    frames = sp.analyse(loaded.samples, loaded.sample_rate, frame_ms)
    if frames.n_frames == 0:
        raise ValueError(
            "%s is %.3f s long, shorter than one %g ms frame, so it holds no spectrum "
            "to learn from" % (label, loaded.duration, frame_ms)
        )
    result = profile_from_frames(frames)
    if result is None:
        raise ValueError(
            "%s is digital silence throughout, so it holds no spectrum to learn from"
            % label
        )
    level, spread = result
    impact_db, impact_spread_db, impact_count = impact_envelope(frames, sensitivity)
    return Profile(
        centre_hz=frames.layout.centre_hz.copy(),
        level_db=level,
        spread_db=spread,
        origin="recording",
        sample_rate=loaded.sample_rate,
        frame_ms=float(frame_ms),
        duration_s=loaded.duration,
        source=loaded.source,
        notes=list(loaded.notes),
        warnings=list(loaded.warnings),
        impact_db=impact_db,
        impact_spread_db=impact_spread_db,
        impact_count=impact_count,
    )


def _widths_from_centres(centres: np.ndarray) -> np.ndarray:
    """Band widths implied by log-spaced centres (edges at geometric midpoints)."""
    c = np.asarray(centres, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] * 0.5])
    inner = np.sqrt(c[:-1] * c[1:])
    edges = np.concatenate(([c[0] ** 2 / inner[0]], inner, [c[-1] ** 2 / inner[-1]]))
    return np.diff(edges)


def map_profile(
    profile: Profile,
    frames: sp.Frames,
    rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Put a profile onto this recording's bands.

    Returns ``(level_db, spread_db, comparable, notes)``; ``comparable`` marks
    the bands the reference actually covers.
    """
    notes: List[str] = []
    layout = frames.layout
    test_c = layout.centre_hz
    ref_c = profile.centre_hz
    nb = test_c.size
    level = np.zeros(nb)
    spread = np.zeros(nb)
    comparable = np.zeros(nb, dtype=bool)

    idx = np.clip(np.searchsorted(ref_c, test_c), 0, ref_c.size - 1)
    best = idx.copy()
    left = np.clip(idx - 1, 0, ref_c.size - 1)
    closer_left = np.abs(np.log(ref_c[left] / test_c)) < np.abs(np.log(ref_c[idx] / test_c))
    best[closer_left] = left[closer_left]
    exact = np.abs(np.log(ref_c[best] / test_c)) < 0.01
    in_range = (test_c >= ref_c[0] / 1.02) & (test_c <= ref_c[-1] * 1.02)
    aligned = bool(np.all(exact[in_range])) and in_range.any()

    if aligned:
        comparable = exact & in_range
        level[comparable] = profile.level_db[best[comparable]]
        spread[comparable] = profile.spread_db[best[comparable]]
    elif ref_c.size >= 2:
        # Different frame_ms (or a hand-made profile): interpolate the power
        # density in log-frequency, then scale back up to this layout's bands.
        comparable = in_range
        ref_density = profile.level_db - 10.0 * np.log10(_widths_from_centres(ref_c))
        density = np.interp(np.log(test_c), np.log(ref_c), ref_density)
        level = density + 10.0 * np.log10(layout.width_hz)
        spread = np.interp(np.log(test_c), np.log(ref_c), profile.spread_db)
        notes.append(
            "the reference profile's %d bands do not line up with this recording's %d "
            "(it was made with a different frame_ms), so it was interpolated onto "
            "them; treat small differences with caution" % (ref_c.size, nb)
        )
    else:
        comparable = exact & in_range
        level[comparable] = profile.level_db[best[comparable]]
        spread[comparable] = profile.spread_db[best[comparable]]

    if profile.origin == "recording" and profile.sample_rate and profile.sample_rate != rate:
        notes.append(
            "the reference was recorded at %d Hz and this recording at %d Hz; the "
            "reference's spectrum was resampled onto this recording's bands"
            % (profile.sample_rate, rate)
        )
    missing = ~comparable
    if missing.any() and comparable.any():
        top = layout.high_hz[comparable][-1]
        notes.append(
            "%d band%s above %.0f Hz could not be compared, because the reference "
            "does not reach that high" % (int(missing.sum()), "" if missing.sum() == 1 else "s", top)
        )
    return level, spread, comparable, notes


# --------------------------------------------------------------------------- helpers


def _peak_frequency(
    samples: np.ndarray,
    frames: sp.Frames,
    frame_idx: np.ndarray,
    baseline_idx: np.ndarray,
    band: int,
) -> float:
    """Frequency of the strongest new narrowband energy in ``band`` over some frames."""
    layout = frames.layout
    n_bins = frames.frame_len // 2 + 1
    # Only the band's own bins: the frequency named must lie in the band named.
    lo = max(1, int(layout.starts[band]))
    hi = max(lo + 1, min(n_bins, int(layout.stops[band])))
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frames.frame_len) / frames.frame_len)

    def mean_power(idx: np.ndarray) -> np.ndarray:
        pick = idx[np.linspace(0, idx.size - 1, min(idx.size, 24)).astype(np.int64)]
        acc = np.zeros(hi - lo)
        for i in pick:
            seg = samples[frames.starts[i] : frames.starts[i] + frames.frame_len] * win
            spec = np.fft.rfft(seg)[lo:hi]
            acc += spec.real ** 2 + spec.imag ** 2
        return acc / max(1, pick.size)

    event_power = mean_power(frame_idx)
    target = event_power
    if baseline_idx.size:
        excess = event_power - mean_power(baseline_idx)
        if np.any(excess > 0):
            target = excess
    k = int(np.argmax(target))
    offset = 0.0
    if 0 < k < target.size - 1:
        a, b, c = np.log(np.maximum(target[k - 1 : k + 2], 1e-30))
        denom = a - 2.0 * b + c
        if denom < 0:
            offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    freq = (lo + k + offset) * frames.sample_rate / float(frames.frame_len)
    return float(np.clip(freq, layout.low_hz[band], layout.high_hz[band]))


def _refine_burst(
    samples: np.ndarray, rate: int, t0: float, t1: float, rise_db: float
) -> Tuple[float, float]:
    """Tighten a burst's frame-accurate span to the millisecond blocks that carry it.

    Energy of the first difference - which weights each frequency by itself,
    so a low hum hardly registers while a knock's broadband edge does - is
    measured in 1 ms blocks and compared with the quarter second before the
    event; blocks at least ``max(6, rise_db / 2)`` dB above that mark the burst.
    If nothing clears the bar the frame span is kept.
    """
    block = max(1, rate // 1000)
    i0, i1 = int(round(t0 * rate)), int(round(t1 * rate))
    if i0 < 1:
        return t0, t1
    seg = np.diff(samples[i0 - 1 : i1])
    before = np.diff(samples[max(0, i0 - rate // 4 - 1) : i0])
    if seg.size < 3 * block or before.size < 3 * block:
        return t0, t1
    energy = np.mean(seg[: seg.size // block * block].reshape(-1, block) ** 2, axis=1)
    context = np.mean(before[: before.size // block * block].reshape(-1, block) ** 2, axis=1)
    reference = float(np.median(context))
    if reference <= 0:
        reference = float(np.max(context))
    if reference <= 0:
        return t0, t1
    above = np.flatnonzero(energy > reference * 10.0 ** (max(6.0, rise_db / 2.0) / 10.0))
    if above.size == 0:
        return t0, t1
    return float(i0 + above[0] * block) / rate, float(i0 + (above[-1] + 1) * block) / rate


def _fmt_hz(hz: float) -> str:
    if hz >= 1000.0:
        return "%.2f kHz" % (hz / 1000.0)
    return "%.0f Hz" % hz


def _span_text(start: float, end: float, duration: float) -> str:
    begin = "the start" if start <= 1e-9 else "%.2f s" % start
    finish = "the end" if end >= duration - 1e-9 else "%.2f s" % end
    return "from %s to %s" % (begin, finish)


def _interval_union(spans: List[Tuple[float, float]]) -> float:
    total = 0.0
    current: Optional[List[float]] = None
    for start, end in sorted(spans):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        total += current[1] - current[0]
    return total


# --------------------------------------------------------------------------- core


def _empty_report(
    loaded: LoadedAudio,
    frame_ms: float,
    sensitivity: float,
    mode: str,
    compared_against: str,
    notes: List[str],
    warnings: List[str],
) -> AudioAnomalyReport:
    return AudioAnomalyReport(
        anomalies=[],
        scores=np.zeros(0, dtype=np.float64),
        frame_times=np.zeros(0, dtype=np.float64),
        anomaly_ratio=0.0,
        sample_rate=loaded.sample_rate,
        duration_s=loaded.duration,
        frame_ms=float(frame_ms),
        sensitivity=sensitivity,
        mode=mode,
        compared_against=compared_against,
        source=loaded.source,
        notes=notes,
        warnings=warnings,
    )


def analyse_loaded(
    loaded: LoadedAudio,
    profile: Optional[Profile],
    sensitivity: float,
    frame_ms: float,
    extra_notes: Optional[List[str]] = None,
    extra_warnings: Optional[List[str]] = None,
) -> AudioAnomalyReport:
    """Score one loaded recording against ``profile`` (or itself) and build the report."""
    notes: List[str] = list(loaded.notes) + list(extra_notes or [])
    warnings: List[str] = list(loaded.warnings) + list(extra_warnings or [])
    mode = "self" if profile is None else "reference"
    compared = (
        "the recording's own typical sound (no reference given)"
        if profile is None
        else profile.describe()
    )
    if profile is not None:
        notes.extend(profile.notes)
        warnings.extend(profile.warnings)

    sens = sensitivity
    samples = loaded.samples
    rate = loaded.sample_rate
    frames = sp.analyse(samples, rate, frame_ms)
    n = frames.n_frames
    if n == 0:
        notes.append(
            "the audio is %.3f s long, shorter than one %g ms analysis frame, so there "
            "was nothing to analyse" % (loaded.duration, frame_ms)
        )
        return _empty_report(loaded, frame_ms, sens, mode, compared, notes, warnings)

    duration = frames.duration
    live = frames.live
    n_live = int(live.sum())
    nb = frames.n_bands
    band = frames.band_db
    level = frames.rms_db
    start_s = frames.start_s
    end_s = np.minimum(frames.end_s, duration)
    times = frames.times
    scores = np.zeros(n, dtype=np.float64)
    events: List[Event] = []
    sustained_events: List[Event] = []

    if n_live == 0:
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        peak_db = 20.0 * np.log10(peak) if peak > 0 else float("-inf")
        notes.append(
            "the recording is digital silence throughout (peak %s), so there is no "
            "sound to judge and nothing was flagged"
            % ("0" if peak == 0 else "%.1f dBFS" % peak_db)
        )
        return AudioAnomalyReport(
            anomalies=[],
            scores=scores,
            frame_times=times,
            anomaly_ratio=0.0,
            sample_rate=rate,
            duration_s=duration,
            frame_ms=float(frame_ms),
            sensitivity=sens,
            mode=mode,
            compared_against=compared,
            source=loaded.source,
            notes=notes,
            warnings=warnings,
        )

    top_hz = float(frames.layout.high_hz[-1])
    if top_hz < 0.99 * sp.F_MAX_HZ:
        notes.append(
            "at %d Hz the frequency bands reach up to %s; sound between that and the "
            "%s Nyquist limit is not monitored, so record at a higher sample rate to "
            "watch higher frequencies" % (rate, _fmt_hz(top_hz), _fmt_hz(rate / 2.0))
        )

    # ------------------------------------------------------------ silence map
    live_idx = np.flatnonzero(live)
    first_live, last_live = int(live_idx[0]), int(live_idx[-1])
    mid_silent = ~live
    mid_silent[:first_live] = False
    mid_silent[last_live + 1 :] = False
    mostly_live = n_live >= MIN_LIVE_FRACTION * n
    # Frames that straddle leading or trailing silence are partly silent by
    # construction; a level drop there is the sound starting, not a fault.
    straddle = np.zeros(n, dtype=bool)
    reach = int(np.ceil(frames.frame_len / float(frames.hop)))
    if first_live > 0:
        straddle[first_live : first_live + reach] = True
    if last_live < n - 1:
        straddle[max(0, last_live - reach + 1) : last_live + 1] = True
    # Measured on the samples, not the frames, so the note gives the true length.
    audible = np.flatnonzero(np.abs(samples) > 10.0 ** (sp.SILENCE_DBFS / 20.0))
    lead_s = audible[0] / float(rate) if first_live > 0 and audible.size else 0.0
    trail_s = (samples.size - 1 - audible[-1]) / float(rate) if last_live < n - 1 and audible.size else 0.0
    if lead_s > 0 or trail_s > 0:
        pieces = []
        if lead_s > 0:
            pieces.append("the first %.2f s" % lead_s)
        if trail_s > 0:
            pieces.append("the last %.2f s" % trail_s)
        notes.append(
            "%s %s digital silence, taken as the recording starting or stopping "
            "rather than as a dropout" % (" and ".join(pieces), "is" if len(pieces) == 1 else "are")
        )
    if not mostly_live:
        notes.append(
            "only %.0f%% of the recording is sound; the rest is digital silence, which "
            "was taken as its normal state, so silent gaps are not reported as "
            "dropouts" % (100.0 * n_live / n)
        )

    hop_s = frames.hop_seconds
    # Smoothing runs over the frames of sound; the context either side of a
    # frame spans the whole recording, silence included.
    width = _sustain_frames(hop_s, n_live)
    side = 2 * _sustain_frames(hop_s, n)
    k_broad = min(nb, max(3, nb // 4))
    tonal_max = max(3, nb // 8)

    # ------------------------------------------------------------ short rises
    # Spreads measured on few frames are themselves uncertain; they are widened
    # by one standard error of the MAD so short recordings are not trigger-happy.
    widen_fast = _small_sample_factor(n_live if mostly_live else n)
    widen_slow = _small_sample_factor(n_live / float(width))
    rise_fast, sigma_band = _fast_rise(frames)
    # A known-good machine that makes impacts of its own (a press, a pump's
    # valves) sets how strong a normal one is, band by band; a short sound is
    # then judged by how far it goes beyond that.
    impact = np.zeros(nb)
    sigma_short = sigma_band
    if profile is not None and profile.impact_db is not None:
        centres = frames.layout.centre_hz
        impact, matched = _by_centre(profile.centre_hz, profile.impact_db, centres)
        impact_sigma, _ = _by_centre(profile.centre_hz, profile.impact_spread_db, centres)
        # This recording's own band spread is inflated by the very impacts that
        # are normal here; how much they vary in the reference is the yardstick.
        sigma_short = np.where(matched, impact_sigma, sigma_band)
        per_s = profile.impact_count / profile.duration_s if profile.duration_s else 0.0
        notes.append(
            "the reference itself holds %d short sounds (impacts, clicks), about %.1f "
            "per second; they were taken as part of the machine's normal sound, so a "
            "short sound is reported only when it clearly exceeds them"
            % (profile.impact_count, per_s)
        )
    s_fast, burst_score, narrow_score = _short_event_scores(rise_fast - impact, sigma_short, live)
    burst_frames = burst_score >= sens
    narrow_frames = (narrow_score >= sens) & ~burst_frames

    # ------------------------------------------------------------ sustained rises
    rise_slow = np.zeros((n, nb))
    s_slow = np.zeros((n, nb))
    s_tone = np.zeros((n, nb))
    tone_up = np.zeros((n, nb), dtype=bool)
    broad_up = np.zeros(n, dtype=bool)
    tone_frame_score = np.zeros(n)
    broad_frame_score = np.zeros(n)
    comb_drop_db = np.zeros(n)
    d_slow = np.zeros(n)
    slow_drop_up = np.zeros(n, dtype=bool)
    smooth = sp.window_median(band[live], -(width // 2), width)
    # However steady the smoothed levels look, they cannot be steadier than the
    # frame-to-frame noise averaged over the window allows (half the frames in a
    # window are independent, and a median is 1.25x noisier than a mean).
    sigma_theory = 1.25 * sigma_band / np.sqrt(max(1.0, width / 2.0))
    floor = np.full(nb, MIN_SPREAD_DB)
    lag_slow = widen_slow * sp.lag_spread(smooth, width)
    can_self = n_live >= MIN_LIVE_FRAMES_FOR_SUSTAINED
    # The recording's own normal: the quieter three quarters of each band. The
    # spread of its lower tail (5th to 25th percentile) measures slow natural
    # wander, and energy that appears for up to three quarters of the recording
    # can inflate neither.
    q05, self_base = np.percentile(smooth, [5.0, RISE_BASELINE_PERCENTILE], axis=0)
    self_sigma = np.maximum.reduce(
        [lag_slow, widen_slow * (self_base - q05) / LOWER_TAIL_TO_SIGMA, sigma_theory, floor]
    )
    if profile is None:
        comparable = np.zeros(nb, dtype=bool)
        base_slow, sigma_slow = self_base, self_sigma
        judged = np.full(nb, can_self)
        if not can_self:
            notes.append(
                "only %.2f s of sound, too little to learn a typical sound from, so "
                "only short events were looked for; give a reference to judge "
                "sustained changes" % (n_live * hop_s)
            )
    else:
        ref_level, ref_spread, comparable, map_notes = map_profile(profile, frames, rate)
        notes.extend(map_notes)
        ref_sigma = np.maximum.reduce([ref_spread, lag_slow, sigma_theory, floor])
        base_slow = np.where(comparable, ref_level, self_base)
        sigma_slow = np.where(comparable, ref_sigma, self_sigma)
        judged = comparable | can_self
        if not comparable.any():
            warnings.append(
                "the reference covers none of this recording's frequency bands, so the "
                "recording was judged against its own typical sound instead"
            )
        elif (~comparable).any() and can_self:
            notes.append(
                "the bands the reference does not cover were judged against this "
                "recording's own typical sound instead"
            )
        if comparable.any():
            offset = float(np.median(smooth[:, comparable] - ref_level[comparable]))
            if abs(offset) >= GAIN_NOTE_DB:
                notes.append(
                    "this recording is %.1f dB %s than the reference on average across "
                    "the spectrum; if the microphone, its gain or its position changed, "
                    "record a new reference" % (abs(offset), "louder" if offset > 0 else "quieter")
                )
    # A smoothed band hovering at the threshold would cross it now and then by
    # chance. A sustained change therefore has to hold at or above it for half
    # the smoothing window - as long as the smoothing needs to register a real
    # change at all - and then lasts while it stays above SUSTAIN_HYSTERESIS of it.
    min_hold = max(1, width // 2)
    low = SUSTAIN_HYSTERESIS * sens
    if judged.any():
        live_rise = smooth - base_slow
        live_rise[:, ~judged] = 0.0
        live_score = _score(live_rise, sigma_slow)
        live_score[:, ~judged] = 0.0
        # A tone is a band standing out from the rest. Whatever rise the whole
        # spectrum shares - a gain change, the machine running harder - is a
        # level change, so tones are judged above it. The median cannot be moved
        # by the few bands a tone occupies once there are enough bands.
        if int(judged.sum()) > 2 * tonal_max:
            common = np.maximum(np.median(live_rise[:, judged], axis=1), 0.0)
        else:
            common = np.zeros(n_live)
        tone_score = _score(live_rise - common[:, None], sigma_slow)
        tone_score[:, ~judged] = 0.0
        # Broadband: the score that more than tonal_max bands reach together.
        if nb > tonal_max:
            broad_score = np.sort(live_score, axis=1)[:, -(tonal_max + 1)]
        else:
            broad_score = np.zeros(n_live)
        tone_held = _held(tone_score, min_hold)
        broad_held = _held(broad_score, min_hold)
        rise_slow[live] = live_rise
        s_slow[live] = live_score
        s_tone[live] = tone_score
        tone_up[live] = _hold(tone_held >= sens, tone_score >= low)
        broad_up[live] = _hold(broad_held >= sens, broad_score >= low)
        tone_frame_score[live] = tone_held.max(axis=1)
        broad_frame_score[live] = broad_held
    # The overall level, for sustained drops: against the reference's level over
    # the bands it covers, or else against the louder quarter of the recording.
    comb: Optional[np.ndarray] = None
    if comparable.any():
        comb = sp.combined_db(smooth[:, comparable])
        base_comb = float(sp.combined_db(ref_level[comparable]))
    elif can_self:
        comb = sp.combined_db(smooth)
        base_comb = float(np.percentile(comb, DROP_BASELINE_PERCENTILE))
    if comb is not None:
        sigma_comb = max(widen_slow * float(sp.lag_spread(comb, width)), MIN_SPREAD_DB)
        drop_c = base_comb - comb
        comb_drop_db[live] = drop_c
        # Held and with hysteresis, exactly like a sustained rise.
        drop_live = _score(drop_c, np.full(drop_c.shape, sigma_comb))
        drop_held = _held(drop_live, min_hold)
        d_slow[live] = drop_held
        slow_drop_up[live] = _hold(drop_held >= sens, drop_live >= low)
    slow_score = np.maximum(tone_frame_score, broad_frame_score)
    tone_count = tone_up.sum(axis=1)
    tone_frames = (tone_count >= 1) & (tone_count <= tonal_max)
    # More bands standing out than a tone can fill is a change of shape across
    # the spectrum (a hiss, a rougher sound), reported as a level rise.
    rise_frames = broad_up | (tone_count > tonal_max)
    slow_frames = tone_frames | rise_frames

    # ------------------------------------------------------------ level drops
    sigma_level = widen_fast * max(
        float(sp.residual_spread(level[live], width)) if mostly_live else 0.0,
        float(sp.lag_spread(level[live], 2)) if mostly_live else 0.0,
        MIN_SPREAD_DB,
    )
    lvl_before = sp.window_median(level, -(GAP_FRAMES + side), side)
    lvl_after = sp.window_median(level, GAP_FRAMES + 1, side)
    fast_drop_db = np.minimum(lvl_before, lvl_after) - level
    d_fast = _score(fast_drop_db, np.full(n, sigma_level))
    d_fast[~live | straddle] = 0.0
    d_slow[straddle] = 0.0
    slow_drop_up[straddle] = False
    normal_level = float(np.median(level[live]))
    d_silent = np.zeros(n)
    if mostly_live:
        d_silent[mid_silent] = max(sens, (normal_level - sp.SILENCE_DBFS) / DB_PER_UNIT)

    # Exact-zero holes inside the sound, too short to empty a whole frame.
    zero_spans: List[Tuple[int, int]] = []
    d_zero = np.zeros(n)
    if mostly_live and samples.size:
        min_zero = max(4, int(round(ZERO_RUN_S * rate)))
        ends_of = frames.starts + frames.frame_len
        for a, b in find_runs(samples == 0.0):
            if b - a < min_zero or a == 0 or b >= samples.size:
                continue
            i0 = int(np.searchsorted(ends_of, a, side="right"))
            i1 = int(np.searchsorted(frames.starts, b, side="left"))
            if i1 <= i0:
                continue
            lo_ctx, hi_ctx = max(0, i0 - side), min(n, i1 + side)
            context = level[lo_ctx:hi_ctx][live[lo_ctx:hi_ctx]]
            if context.size == 0:
                continue
            context_db = float(np.median(context))
            if context_db < ZERO_RUN_CONTEXT_DBFS:
                continue
            zero_spans.append((a, b))
            value = max(sens, (context_db - sp.SILENCE_DBFS) / DB_PER_UNIT)
            d_zero[i0:i1] = np.maximum(d_zero[i0:i1], value)
    drop_score = np.maximum.reduce([d_fast, d_slow, d_silent, d_zero])
    drop_frames = (drop_score >= sens) | slow_drop_up
    drop_db = np.maximum(fast_drop_db, comb_drop_db)
    drop_db[~live] = np.maximum(drop_db[~live], normal_level - level[~live])

    # ------------------------------------------------------------ clipping
    clip_frame = np.zeros(n)
    clip_spans: List[Tuple[int, int]] = []
    if loaded.clipped is not None and loaded.clipped.any():
        csum = np.concatenate(([0], np.cumsum(loaded.clipped, dtype=np.int64)))
        per_frame = csum[frames.starts + frames.frame_len] - csum[frames.starts]
        frac = per_frame / float(frames.frame_len)
        clip_frame = np.where(per_frame > 0, sens + 100.0 * frac, 0.0)
        clip_spans = merge_runs(find_runs(loaded.clipped), frames.frame_len)

    scores = np.maximum.reduce(
        [burst_score, narrow_score, slow_score, drop_score, clip_frame]
    )

    # ------------------------------------------------------------ events
    unflagged = live & ~burst_frames & ~narrow_frames & ~slow_frames & ~drop_frames

    burst_runs = merge_runs(find_runs(burst_frames), GAP_FRAMES)
    narrow_runs = merge_runs(find_runs(narrow_frames), GAP_FRAMES)
    tonal_runs = merge_runs(list(narrow_runs) + find_runs(tone_frames), GAP_FRAMES)
    rise_runs = merge_runs(find_runs(rise_frames), GAP_FRAMES)

    # A short broadband rise seen by both views is one burst, not two events.
    kept_rise: List[Tuple[int, int]] = []
    for a, b in rise_runs:
        overlaps = [r for r in burst_runs if r[0] < b and a < r[1]]
        if overlaps and end_s[b - 1] - start_s[a] < BURST_MAX_S:
            burst_runs.append((a, b))
        else:
            kept_rise.append((a, b))
    burst_runs = merge_runs(burst_runs, GAP_FRAMES)

    for a, b in burst_runs:
        seg = burst_score[a:b]
        peak = a + int(np.argmax(seg))
        top = np.argsort(s_fast[peak])[-k_broad:]
        rise_db = float(np.mean(np.maximum(rise_fast[peak, top], 0.0)))
        n_up = int(np.sum(s_fast[peak] >= sens))
        t0, t1 = _refine_burst(samples, rate, float(start_s[a]), float(end_s[b - 1]), rise_db)
        if impact.any():
            beyond = float(np.mean(np.maximum(rise_fast[peak, top] - impact[top], 0.0)))
            message = (
                "Broadband burst (an impact, knock or click) at %.3f s lasting %.3f s: "
                "%.1f dB above the surrounding sound, and %.1f dB beyond the reference's "
                "own impacts in %d of %d bands."
                % (t0, t1 - t0, rise_db, beyond, max(n_up, 1), nb)
            )
        else:
            message = (
                "Broadband burst (an impact, knock or click) at %.3f s lasting %.3f s: "
                "%.1f dB above the surrounding sound in %d of %d bands."
                % (t0, t1 - t0, rise_db, max(n_up, 1), nb)
            )
        events.append(Event(t0, t1, "burst", float(np.max(seg)), message))

    tonal_weight = np.maximum(s_fast * narrow_frames[:, None] / NARROW_MARGIN, s_tone * tone_up)
    baseline = np.flatnonzero(unflagged)
    for a, b in tonal_runs:
        which = int(np.argmax(tonal_weight[a:b].sum(axis=0)))
        sustained = bool(tone_frames[a:b].any())
        in_run = np.arange(a, b)
        in_run = in_run[live[in_run]]
        if in_run.size == 0:
            continue
        if sustained:
            rise_db = float(np.median(rise_slow[in_run, which]))
        else:
            rise_db = float(np.max(rise_fast[in_run, which]))
        freq = _peak_frequency(samples, frames, in_run, baseline, which)
        score = float(np.max(np.maximum(narrow_score[a:b], tone_frame_score[a:b])))
        if sustained:
            message = (
                "New tone near %s %s: %.1f dB above normal in the %s-%s band."
                % (
                    _fmt_hz(freq),
                    _span_text(start_s[a], end_s[b - 1], duration),
                    rise_db,
                    _fmt_hz(frames.layout.low_hz[which]),
                    _fmt_hz(frames.layout.high_hz[which]),
                )
            )
        else:
            message = (
                "Short tonal sound near %s at %.2f s lasting %.2f s: %.1f dB above the "
                "surrounding sound." % (_fmt_hz(freq), start_s[a], end_s[b - 1] - start_s[a], rise_db)
            )
        event = Event(float(start_s[a]), float(end_s[b - 1]), "tonal", score, message)
        events.append(event)
        if sustained:
            sustained_events.append(event)

    for a, b in kept_rise:
        seg = np.arange(a, b)
        seg = seg[live[seg]]
        if seg.size == 0:
            continue
        # How far each band sat above normal over the whole event, and which
        # bands that makes risen.
        band_rise = np.median(rise_slow[seg], axis=0)
        band_up = _score(band_rise, sigma_slow)
        up = band_up >= sens
        if not up.any():
            up = band_up >= low
        if not up.any():
            up = band_rise >= band_rise.max()
        events.append(
            Event(
                float(start_s[a]),
                float(end_s[b - 1]),
                "level_rise",
                float(np.max(slow_score[a:b])),
                "Sound %.1f dB louder than normal across %d of %d bands %s."
                % (
                    float(np.mean(band_rise[up])),
                    int(up.sum()),
                    nb,
                    _span_text(start_s[a], end_s[b - 1], duration),
                ),
            )
        )

    # Drops: frame runs, plus each zero hole on its own exact span.
    frame_drop = (np.maximum.reduce([d_fast, d_silent]) >= sens) | slow_drop_up
    drop_runs = merge_runs(find_runs(frame_drop), GAP_FRAMES)
    spans: List[Tuple[float, float, bool, List[Tuple[int, int]]]] = []
    for a, b in drop_runs:
        holes = [z for z in zero_spans if z[0] < frames.starts[b - 1] + frames.frame_len and frames.starts[a] < z[1]]
        spans.append((float(start_s[a]), float(end_s[b - 1]), True, holes))
    for z in zero_spans:
        t0, t1 = z[0] / float(rate), z[1] / float(rate)
        if not any(s[0] <= t0 and t1 <= s[1] for s in spans if s[2]):
            spans.append((t0, t1, False, [z]))
    spans.sort()
    merged_spans: List[List[Any]] = []
    for s in spans:
        if merged_spans and s[0] <= merged_spans[-1][1]:
            last = merged_spans[-1]
            last[1] = max(last[1], s[1])
            last[2] = last[2] or s[2]
            last[3] = last[3] + [h for h in s[3] if h not in last[3]]
        else:
            merged_spans.append([s[0], s[1], s[2], list(s[3])])
    for t0, t1, from_frames, holes in merged_spans:
        idx = np.flatnonzero((end_s > t0) & (start_s < t1))
        if idx.size == 0:
            continue
        score = float(np.max(drop_score[idx]))
        worst_db = float(np.max(drop_db[idx])) if from_frames else 0.0
        floor_db = float(np.min(level[idx]))
        silent_inside = bool(np.any(mid_silent[idx] & (d_silent[idx] > 0)))
        hole_ms = 1000.0 * sum(h[1] - h[0] for h in holes) / rate
        if silent_inside or holes or worst_db >= DROPOUT_DB:
            if from_frames and floor_db < sp.SILENCE_DBFS:
                message = "Signal dropped out %s: the sound fell to digital silence." % (
                    _span_text(t0, t1, duration)
                )
            elif from_frames:
                message = (
                    "Signal dropped out %s: the level fell to %.0f dBFS, %.0f dB below "
                    "normal." % (_span_text(t0, t1, duration), floor_db, normal_level - floor_db)
                )
                if holes:
                    message = message[:-1] + ", with %.0f ms of exact digital silence." % hole_ms
            else:
                message = (
                    "Signal cut out at %.3f s: %.0f ms of exact digital silence inside the "
                    "sound, typically a dropped buffer or a bad connection." % (t0, hole_ms)
                )
            events.append(Event(t0, t1, "dropout", score, message))
        else:
            events.append(
                Event(
                    t0,
                    t1,
                    "level_drop",
                    score,
                    "Level fell %.1f dB below normal (down to %.1f dBFS) %s."
                    % (worst_db, floor_db, _span_text(t0, t1, duration)),
                )
            )

    for a, b in clip_spans:
        count = int(np.count_nonzero(loaded.clipped[a:b]))
        span = max(1, b - a)
        idx = np.flatnonzero((frames.starts < b) & (frames.starts + frames.frame_len > a))
        score = float(np.max(clip_frame[idx])) if idx.size else sens + 100.0 * count / span
        t0, t1 = a / float(rate), b / float(rate)
        events.append(
            Event(
                t0,
                t1,
                "clipping",
                max(score, sens),
                "Clipping from %.3f s to %.3f s: %d sample%s (%.1f%%) sit at or beyond "
                "full scale, the %s."
                % (t0, t1, count, "" if count == 1 else "s", 100.0 * count / span, loaded.rail.label),
            )
        )

    # An abrupt step - a tone switching on, a level falling off a cliff, a hole
    # in the signal - splatters energy across the spectrum for a frame or two.
    # Those splashes sit on the step's edges and are part of it, not a knock.
    sustained_events.extend(e for e in events if e.kind in ("level_rise", "level_drop", "dropout"))
    # Sustained edges are known to a frame or two, burst edges to a millisecond.
    edge = 2.0 * frames.frame_len / float(rate)
    # The start and end of the recording are not steps in the sound; the edges
    # of leading and trailing digital silence are.
    steps = [
        t
        for e in sustained_events
        for t in (e.start_s, e.end_s)
        if edge < t < duration - edge
    ]
    if mostly_live:
        if first_live > 0:
            steps.append(float(start_s[first_live]) + frames.frame_len / float(rate))
        if last_live < n - 1:
            steps.append(float(end_s[last_live]) - frames.frame_len / float(rate))

    def on_an_edge(event: Event) -> bool:
        return any(event.start_s <= t + edge and event.end_s >= t - edge for t in steps)

    sustained_ids = {id(e) for e in sustained_events}
    events = [
        e for e in events
        if not (e.kind in ("burst", "tonal") and id(e) not in sustained_ids and on_an_edge(e))
    ]
    events.sort(key=lambda e: (e.start_s, e.end_s, e.kind))
    n_bursts = sum(1 for e in events if e.kind == "burst")
    if profile is None and n_bursts >= MANY_BURSTS:
        notes.append(
            "%d bursts, about %.1f per second; if impacts are part of this machine's "
            "normal sound, pass a healthy recording that contains them as the "
            "reference, and only impacts beyond its own are reported"
            % (n_bursts, n_bursts / duration)
        )
    elif profile is not None and profile.origin == "profile" and n_bursts >= MIN_REFERENCE_IMPACTS:
        notes.append(
            "a saved profile holds only the reference's typical spectrum, not its "
            "impacts; if impacts are part of this machine's normal sound, pass the "
            "reference recording itself so that its impacts are learned too"
        )
    covered = _interval_union([(e.start_s, e.end_s) for e in events])
    ratio = float(min(1.0, covered / duration)) if duration > 0 else 0.0
    if profile is None and can_self and n_live * hop_s < ROUGH_BELOW_S:
        notes.append(
            "the recording holds only %.2f s of sound, so its typical sound was learned "
            "from few frames; treat sustained results as rough, or give a reference"
            % (n_live * hop_s)
        )
    logger.debug(
        "%d frames, %d live, %d bands, width=%d, %d events; component maxima: "
        "burst=%.2f narrow=%.2f sustained=%.2f fast_drop=%.2f slow_drop=%.2f "
        "silent=%.2f zero=%.2f clip=%.2f",
        n, n_live, nb, width, len(events),
        burst_score.max(), narrow_score.max(), slow_score.max(), d_fast.max(),
        d_slow.max(), d_silent.max(), d_zero.max(), clip_frame.max(),
    )
    return AudioAnomalyReport(
        anomalies=events,
        scores=scores,
        frame_times=times,
        anomaly_ratio=ratio,
        sample_rate=rate,
        duration_s=duration,
        frame_ms=float(frame_ms),
        sensitivity=sens,
        mode=mode,
        compared_against=compared,
        source=loaded.source,
        notes=notes,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- public


def _is_bare_array(obj: Any) -> bool:
    """True for array-like audio that carries no sample rate of its own."""
    if obj is None or isinstance(obj, (str, os.PathLike)) or _looks_like_pair(obj):
        return False
    return as_profile(obj) is None


def _reference_rate(reference: Any, sample_rate: Optional[float], fallback: Optional[int]) -> Tuple[Optional[float], List[str]]:
    """The sample rate to read a reference with: only a bare array needs one."""
    if not _is_bare_array(reference):
        return None, []
    if sample_rate is not None:
        return sample_rate, []
    if fallback is None:
        return None, []
    return fallback, [
        "the reference was a bare array, so it was taken to share this "
        "recording's %d Hz sample rate" % fallback
    ]


def _profile_frame_ms(profile: Profile, frame_ms: float) -> Tuple[float, List[str]]:
    """The frame length to analyse with so a saved profile's bands line up."""
    if profile.origin != "profile":
        return frame_ms, []
    made_with = sp.frame_ms_of_centres(profile.centre_hz)
    if made_with is None:
        return frame_ms, []
    profile.frame_ms = made_with
    if abs(made_with - frame_ms) < 1e-6:
        return frame_ms, []
    return made_with, [
        "the saved profile was made with frame_ms=%g, so this recording was analysed "
        "with frame_ms=%g too, keeping the frequency bands aligned" % (made_with, made_with)
    ]


def _length_note(profile: Optional[Profile], loaded: LoadedAudio) -> List[str]:
    """A note when the reference recording and this one differ in length."""
    if profile is None or profile.origin != "recording" or not profile.duration_s:
        return []
    if abs(profile.duration_s - loaded.duration) <= 0.01 * max(loaded.duration, 1e-9):
        return []
    return [
        "the reference is %.2f s long and this recording %.2f s; only the "
        "reference's typical spectrum is used, so the lengths need not match"
        % (profile.duration_s, loaded.duration)
    ]


def detect(
    audio: Any,
    *,
    sample_rate: Optional[float] = None,
    reference: Any = None,
    sensitivity: float = 3.0,
    frame_ms: float = 50,
) -> AudioAnomalyReport:
    """Find unusual sounds in a recording.

    Args:
        audio: a ``.wav`` path, a numpy array (mono, or samples x channels), or a
            ``(samples, sample_rate)`` pair. Stereo is mixed to mono; clipping is
            measured per channel first. The array is never modified.
        sample_rate: samples per second; required when ``audio`` is a bare array.
        reference: a known-good recording (any of the forms ``audio`` accepts)
            or a profile saved from :func:`spectral_profile`. Each frame's
            spectrum is then compared with it. Without one, the recording's own
            typical sound is learned and departures from it are flagged. A
            reference of a different length or sample rate is fine: only its
            typical spectrum is used, resampled onto this recording's bands.
            Impacts a reference recording makes regularly (a press's strokes)
            are learned as normal, so only bursts beyond them are reported.
        sensitivity: the score a frame must reach to be anomalous, in robust
            standard deviations; each unit also needs 2 dB of real change.
            Higher flags less.
        frame_ms: analysis frame length in milliseconds (frames overlap by half).

    Returns:
        An :class:`AudioAnomalyReport`. Call ``.summary()`` for the story.
    """
    sens = _check_sensitivity(sensitivity)
    loaded = load_audio(audio, sample_rate, label="audio")
    sp.frame_plan(loaded.sample_rate, frame_ms)  # a bad frame_ms fails before the reference loads
    notes: List[str] = []
    warnings: List[str] = []
    profile: Optional[Profile] = None
    fms = float(frame_ms)
    if reference is not None:
        ref_rate, rate_notes = _reference_rate(reference, sample_rate, loaded.sample_rate)
        notes.extend(rate_notes)
        try:
            profile = build_profile(reference, ref_rate, fms, "the reference", sens)
        except ValueError as exc:
            if "holds no spectrum" not in str(exc):
                raise
            warnings.append(
                "%s; the recording was compared against its own typical sound instead"
                % exc
            )
        if profile is not None:
            fms, frame_notes = _profile_frame_ms(profile, fms)
            notes.extend(frame_notes)
            if profile.origin == "profile":
                notes.append(
                    "the reference was read as a saved spectral profile of %d bands"
                    % profile.centre_hz.size
                )
            notes.extend(_length_note(profile, loaded))
    return analyse_loaded(loaded, profile, sens, fms, notes, warnings)


def spectral_profile(
    audio: Any,
    *,
    sample_rate: Optional[float] = None,
    frame_ms: float = 50,
) -> np.ndarray:
    """The typical spectrum of a recording, for saving as a reference.

    Args:
        audio: a ``.wav`` path, a numpy array, or ``(samples, sample_rate)``.
        sample_rate: samples per second; required for a bare array.
        frame_ms: analysis frame length; use the same value when checking.

    Returns:
        ``(n_bands, 3)`` float64 array, one row per frequency band, columns
        ``frequency_hz`` (band centre), ``level_db`` (the typical power in that
        band, dBFS - the average spectrum) and ``spread_db`` (how much it
        wanders in this recording). Save it with ``numpy.save`` and pass it back
        as ``reference=``. It holds the spectrum only: for a machine whose
        normal sound includes impacts, pass the reference recording itself.

    Raises:
        ValueError: the recording is shorter than one frame or silent throughout.
    """
    return build_profile(audio, sample_rate, frame_ms, "audio").as_array()


class Monitor:
    """A detector bound to one known-good reference, for checking many recordings.

    The reference is analysed once, up front. Every :meth:`check` then compares
    a new recording with it, exactly as ``detect(audio, reference=...)`` would.

    Args:
        reference: a known-good recording (``.wav`` path, array, or
            ``(samples, sample_rate)``) or a profile from :func:`spectral_profile`.
        sample_rate: samples per second for bare arrays, both the reference and
            any array later passed to :meth:`check`.
        sensitivity: threshold for a frame to count as anomalous (default 3.0).
        frame_ms: analysis frame length in milliseconds (default 50). A saved
            profile carries its own, which then wins, with a note.

    Raises:
        ValueError: the reference is silent or shorter than one frame.
    """

    def __init__(
        self,
        reference: Any,
        *,
        sample_rate: Optional[float] = None,
        sensitivity: float = 3.0,
        frame_ms: float = 50,
    ) -> None:
        if reference is None:
            raise ValueError("Monitor needs a reference: a known-good recording or profile")
        self.sensitivity = _check_sensitivity(sensitivity)
        self.sample_rate = sample_rate
        fms = float(frame_ms)
        if self.sample_rate is not None:
            sp.frame_plan(int(round(float(self.sample_rate))), fms)
        ref_rate, self._notes = _reference_rate(reference, sample_rate, None)
        self._profile = build_profile(reference, ref_rate, fms, "the reference", self.sensitivity)
        self.frame_ms, frame_notes = _profile_frame_ms(self._profile, fms)
        self._notes = self._notes + frame_notes

    @property
    def profile(self) -> np.ndarray:
        """The reference's typical spectrum, as :func:`spectral_profile` returns it."""
        return self._profile.as_array()

    def check(self, audio: Any, *, sample_rate: Optional[float] = None) -> AudioAnomalyReport:
        """Compare one recording with the reference and report what differs."""
        rate = sample_rate if sample_rate is not None else self.sample_rate
        loaded = load_audio(audio, rate, label="audio")
        notes = list(self._notes) + _length_note(self._profile, loaded)
        return analyse_loaded(loaded, self._profile, self.sensitivity, self.frame_ms, notes)

    def __repr__(self) -> str:
        return "Monitor(reference=%s, sensitivity=%g, frame_ms=%g)" % (
            self._profile.describe(),
            self.sensitivity,
            self.frame_ms,
        )
