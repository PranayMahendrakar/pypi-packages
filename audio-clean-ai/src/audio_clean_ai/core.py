"""Spectral gating: learn what the noise sounds like, then turn down only that.

The method, in the order it runs, for each channel on its own:

1. Cut the recording into 32 ms Hann frames at 75% overlap and take each
   frame's spectrum with :func:`numpy.fft.rfft`.
2. Learn the noise profile - the power the noise puts into each frequency bin -
   from the quietest stretch of the recording (or take the one given). The
   per-bin median over the stretch is used, corrected so steady noise reads at
   its mean, which keeps a stray breath or click in the stretch from inflating
   it.
3. Decide how trustworthy that profile is. If the quietest stretch is barely
   quieter than the loudest parts, it is probably not a pause at all, and the
   cut is held to a gentle few dB rather than guessing hard.
4. For every time-frequency bin, measure how far it sits above the profile.
   Bins near the profile are noise and are turned down; bins well above it are
   signal and pass. The decision is smoothed across time and frequency first,
   so isolated bins do not flicker on and off - that flicker is the watery
   "musical noise" that makes cheap noise reduction warble.
5. Optionally keep a floor in the 300-3400 Hz speech band, so quiet speech is
   turned down at most so far and never gated away entirely.
6. Resynthesise by overlap-add, cut to exactly the input length, and turn the
   whole output down (never clip it) in the rare case it would pass full scale.

This is a heuristic signal-processing method, not a model. It removes steady
noise - hum, hiss, fans, air conditioning, room tone - because steady noise is
exactly what a single profile can describe. It does little for noise that moves
the way speech does: music, babble, a second speaker, traffic going by.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import numpy as np

from ._audio import Loaded, load_audio, rail_hits
from ._result import ChannelReport, CleanResult, octave_bands
from ._stft import Stft, frame_sizes, smooth

logger = logging.getLogger(__name__)

SILENT_POWER = 1e-12
"""Mean square below which a frame counts as digital silence (-120 dBFS)."""

TINY = 1e-20
"""Keeps logarithms and ratios finite on exact zeros (-200 dB)."""

MIN_PROFILE_SECONDS = 0.1
"""A quiet stretch shorter than this is too short to learn a profile from."""

HEADROOM = 0.999
"""When the output has to be turned down, its peak lands this far inside full scale."""

MAX_PROFILE_FRAMES = 8192
"""A noise clip longer than this many frames is subsampled evenly for the median."""

LOWEST_SNR_DB = -10.0
"""Below this the signal cannot be told from error in the noise estimate, so the
SNR is reported as unknown (``None``) rather than as a confident small number."""


def _median_bias(count: int) -> float:
    """Expected median of ``count`` unit-mean exponential variables.

    The power in one bin of steady noise is exponentially distributed, so its
    median sits below its mean; dividing by this puts the estimate back at the
    mean. It tends to ``ln 2`` for long stretches and is exactly 1 for one frame.
    """
    if count <= 0:
        return 1.0
    harmonic = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, count + 1))])
    if count % 2:
        half = (count - 1) // 2
        return float(harmonic[count] - harmonic[half])
    half = count // 2
    upper = harmonic[count] - harmonic[half - 1]
    lower = harmonic[count] - harmonic[half]
    return float((upper + lower) / 2.0)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True values."""
    if not np.any(mask):
        return 0
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[0::2]))


def _to_db(power: float) -> float:
    """``10*log10`` with exact zeros held at -200 dB."""
    return 10.0 * math.log10(max(float(power), TINY))


class _Stretch:
    """Where the profile came from and whether to believe it."""

    __slots__ = ("first", "count", "silent", "reliable", "note", "contrast_db")

    def __init__(
        self,
        first: int = 0,
        count: int = 0,
        silent: bool = False,
        reliable: bool = True,
        note: str = "",
        contrast_db: Optional[float] = None,
    ) -> None:
        self.first = first
        self.count = count
        self.silent = silent
        self.reliable = reliable
        self.note = note
        self.contrast_db = contrast_db


class SpectralGate:
    """A spectral noise gate with every knob exposed.

    :func:`clean` is this class with its defaults; reach for the class when you
    want to change the analysis or reuse one configuration across many files.

    Args:
        strength: 0 to 1, how hard to cut. 0 returns the audio untouched; 1 cuts
            noise-only bins by ``max_cut_db``.
        preserve_speech: Keep a floor in the speech band so it is never cut by
            more than ``speech_cut_db``.
        frame_ms: Analysis frame length. 32 ms suits speech; longer frames
            resolve hum more finely but smear onsets.
        profile_seconds: Length of the quiet stretch the profile is learned from.
        max_cut_db: The cut, in dB, applied to a noise-only bin at strength 1.
        speech_band: ``(low_hz, high_hz)`` protected when ``preserve_speech``.
        speech_cut_db: The deepest cut allowed inside the speech band.
        gentle_cut_db: The deepest cut allowed when the profile is unreliable.
        min_contrast_db: How far the loudest parts must rise above the quiet
            stretch for that stretch to be believed to be a pause.
        threshold_db: A bin this far above the profile starts to be let through.
        pass_db: A bin this far above the profile passes untouched.
        time_smoothing_ms: Half-width of the smoothing of the gate across time.
        freq_smoothing_hz: Half-width of the smoothing of the gate across frequency.
        block_frames: Frames processed at a time; bounds memory on long files and
            does not change the result.

    Raises:
        ValueError: Any setting is out of range.
    """

    def __init__(
        self,
        strength: float = 0.8,
        *,
        preserve_speech: bool = True,
        frame_ms: float = 32.0,
        profile_seconds: float = 0.4,
        max_cut_db: float = 30.0,
        speech_band: Tuple[float, float] = (300.0, 3400.0),
        speech_cut_db: float = 15.0,
        gentle_cut_db: float = 6.0,
        min_contrast_db: float = 6.0,
        threshold_db: float = 2.0,
        pass_db: float = 10.0,
        time_smoothing_ms: float = 40.0,
        freq_smoothing_hz: float = 100.0,
        block_frames: int = 1024,
    ) -> None:
        strength = float(strength)
        if not (0.0 <= strength <= 1.0) or math.isnan(strength):
            raise ValueError("strength must be between 0 and 1; got {}".format(strength))
        if not 8.0 <= float(frame_ms) <= 128.0:
            raise ValueError(
                "frame_ms must be between 8 and 128; got {}".format(frame_ms)
            )
        if float(profile_seconds) <= 0.0:
            raise ValueError(
                "profile_seconds must be positive; got {}".format(profile_seconds)
            )
        for name, value in (
            ("max_cut_db", max_cut_db),
            ("speech_cut_db", speech_cut_db),
            ("gentle_cut_db", gentle_cut_db),
            ("min_contrast_db", min_contrast_db),
            ("time_smoothing_ms", time_smoothing_ms),
            ("freq_smoothing_hz", freq_smoothing_hz),
        ):
            if float(value) < 0.0 or math.isnan(float(value)):
                raise ValueError("{} must be 0 or more; got {}".format(name, value))
        if float(max_cut_db) > 120.0:
            raise ValueError("max_cut_db above 120 dB is not a cut, it is a mute")
        low, high = (float(speech_band[0]), float(speech_band[1]))
        if not 0.0 < low < high:
            raise ValueError(
                "speech_band must be (low_hz, high_hz) with 0 < low < high; got "
                "{}".format(speech_band)
            )
        if not float(threshold_db) < float(pass_db):
            raise ValueError(
                "threshold_db ({}) must be below pass_db ({})".format(threshold_db, pass_db)
            )
        if int(block_frames) < 1:
            raise ValueError("block_frames must be at least 1; got {}".format(block_frames))
        self.strength = strength
        self.preserve_speech = bool(preserve_speech)
        self.frame_ms = float(frame_ms)
        self.profile_seconds = float(profile_seconds)
        self.max_cut_db = float(max_cut_db)
        self.speech_band = (low, high)
        self.speech_cut_db = float(speech_cut_db)
        self.gentle_cut_db = float(gentle_cut_db)
        self.min_contrast_db = float(min_contrast_db)
        self.threshold_db = float(threshold_db)
        self.pass_db = float(pass_db)
        self.time_smoothing_ms = float(time_smoothing_ms)
        self.freq_smoothing_hz = float(freq_smoothing_hz)
        self.block_frames = int(block_frames)
        # How far the keep/cut decision looks around each bin before it is made:
        # enough to average out the frame-to-frame flutter of noise power.
        self._decide_frames = 2
        self._decide_bins = 1

    # ----------------------------------------------------------------- profile

    def learn_profile(self, audio: Any, *, sample_rate: Optional[int] = None) -> np.ndarray:
        """Learn a noise profile from a clip that is nothing but the noise.

        Every frame of the clip is used, not just its quietest stretch, since
        the whole clip is noise by contract. Digital silence inside the clip is
        skipped.

        Args:
            audio: A ``.wav`` path, a numpy array, or ``(samples, sample_rate)``.
            sample_rate: Required for a bare array.

        Returns:
            Per-bin noise power: 1-D for a one-channel clip, ``(n_channels,
            n_bins)`` otherwise. ``10*log10(profile.sum())`` is the noise level
            in dB relative to full scale.

        Raises:
            ValueError: The clip is empty or unreadable.
        """
        loaded = load_audio(audio, sample_rate)
        n_samples, n_channels = loaded.frames.shape
        if n_samples == 0:
            raise ValueError("the noise clip is empty; a profile needs some noise to learn")
        n_fft, _hop = frame_sizes(loaded.sample_rate, self.frame_ms)
        stft = Stft(n_samples, n_fft, loaded.sample_rate)
        profiles = np.zeros((n_channels, stft.n_bins))
        for channel in range(n_channels):
            padded = stft.pad(loaded.frames[:, channel])
            levels = self._levels(stft, padded)
            use = stft.full_frames() & (levels > SILENT_POWER)
            if not np.any(use):
                use = levels > SILENT_POWER
                if np.any(use):
                    logger.warning(
                        "%s: the noise clip is shorter than one %d ms frame; the "
                        "profile is a rough guess",
                        loaded.source,
                        int(round(self.frame_ms)),
                    )
            if not np.any(use):
                logger.warning(
                    "%s channel %d is digital silence; its noise profile is zero",
                    loaded.source,
                    channel + 1,
                )
                continue
            chosen = np.flatnonzero(use)
            if chosen.size > MAX_PROFILE_FRAMES:
                picks = np.linspace(0, chosen.size - 1, MAX_PROFILE_FRAMES)
                chosen = chosen[np.round(picks).astype(np.int64)]
            picked = np.zeros(stft.n_frames, dtype=bool)
            picked[chosen] = True
            rows = []
            for first in range(0, stft.n_frames, self.block_frames):
                stop = min(stft.n_frames, first + self.block_frames)
                if np.any(picked[first:stop]):
                    power = stft.power(stft.analyse(padded, first, stop))
                    rows.append(power[picked[first:stop]])
            power = np.vstack(rows)
            profiles[channel] = np.median(power, axis=0) / _median_bias(power.shape[0])
        return profiles[0].copy() if n_channels == 1 else profiles

    # ------------------------------------------------------------------- clean

    def clean(
        self,
        audio: Any,
        *,
        sample_rate: Optional[int] = None,
        noise_profile: Optional[np.ndarray] = None,
    ) -> CleanResult:
        """Remove steady background noise from a recording.

        Args:
            audio: A ``.wav`` path, a numpy array (1-D, or 2-D in either
                orientation), or ``(samples, sample_rate)``. Never modified.
            sample_rate: Required for a bare array.
            noise_profile: A profile from :func:`noise_profile` or a previous
                result, used instead of learning one from the recording. 1-D
                applies to every channel; 2-D gives one row per channel.

        Returns:
            A :class:`CleanResult` whose ``audio`` has exactly the input's shape.

        Raises:
            ValueError: The input, the settings or the profile do not fit.
        """
        loaded = load_audio(audio, sample_rate)
        frames = loaded.frames
        n_samples, n_channels = frames.shape
        rate = loaded.sample_rate
        n_fft, _hop = frame_sizes(rate, self.frame_ms)
        n_bins = n_fft // 2 + 1
        frequencies = np.arange(n_bins) * (rate / float(n_fft))
        given = self._check_profile(noise_profile, n_channels, n_bins, rate)

        warnings = []  # type: List[str]
        notes = list(loaded.notes)
        warnings.extend(note for note in loaded.notes if "non-finite" in note)
        output = np.zeros_like(frames)
        profiles = np.zeros((n_channels, n_bins))
        reports = []  # type: List[ChannelReport]
        totals = np.zeros(4)  # noise in, noise out, signal in, signal out

        if n_samples == 0:
            warnings.append("the recording is empty; there was nothing to clean")
            for channel in range(n_channels):
                reports.append(self._silent_report(channel))
        else:
            stft = Stft(n_samples, n_fft, rate)
            def run(channel: int) -> Tuple[np.ndarray, ChannelReport, np.ndarray, np.ndarray]:
                row = given[channel] if given is not None else None
                return self._clean_channel(frames[:, channel], stft, row, channel)

            if n_channels > 1:
                # Channels are independent and numpy releases the GIL in its
                # FFTs and array arithmetic, so threads cut stereo time nearly
                # in half. Results are collected in channel order: deterministic.
                with ThreadPoolExecutor(max_workers=min(n_channels, 8)) as pool:
                    results = list(pool.map(run, range(n_channels)))
            else:
                results = [run(0)]
            for channel, (cleaned, report, profile, sums) in enumerate(results):
                output[:, channel] = cleaned
                profiles[channel] = profile
                reports.append(report)
                totals += sums

        if self.strength == 0.0:
            output = frames.copy()
            notes.append("strength is 0, so the audio was returned unchanged")

        output, gain_db = self._fit_inside_rails(output, loaded)
        before = rail_hits(frames, loaded.positive_rail, loaded.negative_rail)
        after = rail_hits(output, loaded.positive_rail, loaded.negative_rail)
        beyond = rail_hits(frames, loaded.positive_rail + 1e-6, loaded.negative_rail - 1e-6)
        if sum(beyond):
            # Only a float input can go past full scale. That is a level, not
            # clipping: the samples are all there, just too big.
            notes.append(
                "{} input samples are above full scale (peak {:.3g}); a float "
                "recording louder than -1 to 1 is allowed, and the output is "
                "brought back inside it".format(
                    sum(beyond), float(np.max(np.abs(frames)))
                )
            )
        elif sum(before) >= 10:
            warnings.append(
                "{} input samples already sit at full scale ({} at the positive "
                "rail, {} at the negative): the recording was clipped before it "
                "got here, and noise reduction cannot undo that".format(
                    sum(before), before[0], before[1]
                )
            )
        if gain_db < 0.0:
            notes.append(
                "the output was turned down {:.2f} dB so it stays inside full scale "
                "instead of clipping".format(-gain_db)
            )

        live = [report for report in reports if not report.silent]
        if n_samples and not live:
            warnings.append(
                "the recording is digital silence; there is no noise to learn and "
                "nothing was changed"
            )
        for report in live:
            if not report.profile_reliable:
                prefix = (
                    "" if n_channels == 1 else "channel {}: ".format(report.channel + 1)
                )
                warnings.append(prefix + report.reliability_note)
        if live and all(r.profile_reliable for r in live):
            floors = [r.noise_floor_dbfs for r in live if r.noise_floor_dbfs is not None]
            if floors and max(floors) < -75.0:
                notes.append(
                    "the noise floor is already very low ({:.0f} dBFS); there was "
                    "little to remove".format(max(floors))
                )

        noise_in, noise_out, signal_in, signal_out = (float(v) for v in totals)
        reduction = _to_db(noise_in) - _to_db(noise_out) if noise_in > 0.0 else 0.0
        return CleanResult(
            audio=loaded.restore(output),
            sample_rate=rate,
            noise_reduction_db=float(max(reduction, 0.0)),
            noise_profile=profiles[0].copy() if n_channels == 1 else profiles,
            snr_before=_snr(signal_in, noise_in),
            snr_after=_snr(signal_out, noise_out),
            profile_reliable=bool(live) and all(r.profile_reliable for r in live),
            strength=self.strength,
            preserve_speech=self.preserve_speech,
            frequencies=frequencies,
            channels=reports,
            source=loaded.source,
            n_samples=int(n_samples),
            n_channels=int(n_channels),
            channel_axis=loaded.channel_axis,
            output_gain_db=float(gain_db),
            clipping_before={"positive": before[0], "negative": before[1]},
            clipping_after={"positive": after[0], "negative": after[1]},
            positive_rail=float(loaded.positive_rail),
            negative_rail=float(loaded.negative_rail),
            source_bits=loaded.bits,
            warnings=warnings,
            notes=notes,
        )

    # ------------------------------------------------------------- internals

    def _check_profile(
        self, profile: Optional[np.ndarray], n_channels: int, n_bins: int, rate: int
    ) -> Optional[np.ndarray]:
        """Validate a caller's profile and spread it to one row per channel."""
        if profile is None:
            return None
        array = np.array(profile, dtype=np.float64, copy=True)
        if array.ndim == 1:
            array = np.tile(array, (n_channels, 1))
        elif array.ndim == 2 and array.shape[0] == 1:
            array = np.tile(array[0], (n_channels, 1))
        elif array.ndim != 2 or array.shape[0] != n_channels:
            raise ValueError(
                "noise_profile has shape {}; give one row per channel ({} here) or a "
                "single 1-D profile for all of them".format(array.shape, n_channels)
            )
        if array.shape[1] != n_bins:
            raise ValueError(
                "noise_profile has {} bins, but audio at {} Hz with {:g} ms frames is "
                "analysed in {} bins. Learn the profile from a clip at the same "
                "sample rate: noise_profile(clip, sample_rate={})".format(
                    array.shape[1], rate, self.frame_ms, n_bins, rate
                )
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(
                "noise_profile must be finite and non-negative: it is power per bin"
            )
        return array

    def _levels(self, stft: Stft, padded: np.ndarray) -> np.ndarray:
        """Mean square of every frame, one entry per frame."""
        levels = np.empty(stft.n_frames)
        for first in range(0, stft.n_frames, self.block_frames):
            stop = min(stft.n_frames, first + self.block_frames)
            levels[first:stop] = stft.power(stft.analyse(padded, first, stop)).sum(axis=1)
        return levels

    def _find_stretch(self, stft: Stft, levels: np.ndarray) -> _Stretch:
        """Find the quietest stretch of sound and judge whether it is a pause."""
        sounding = levels > SILENT_POWER
        if not np.any(sounding):
            return _Stretch(silent=True, reliable=False, note="the channel is digital silence")
        candidates = stft.full_frames() & sounding
        partial = False
        if not np.any(candidates):
            candidates = sounding
            partial = True
        want = max(1, int(round(self.profile_seconds * stft.sample_rate / stft.hop)))
        # Never let "the quietest stretch" be most of the recording: then it is
        # not a pause, it is the recording.
        count = max(1, min(want, _longest_run(candidates), int(candidates.sum()) // 4))
        db = 10.0 * np.log10(levels + TINY)
        ok = np.concatenate([[0], np.cumsum(candidates)])
        total = np.concatenate([[0.0], np.cumsum(np.where(candidates, db, 0.0))])
        means = (total[count:] - total[:-count]) / count
        means[(ok[count:] - ok[:-count]) != count] = np.inf
        first = int(np.argmin(means))
        stretch_db = float(means[first])
        loud_db = float(np.percentile(db[candidates], 99))
        contrast = loud_db - stretch_db
        seconds = count * stft.hop / float(stft.sample_rate)
        start_s, end_s = stft.frame_seconds(first, first + count - 1)
        stretch = _Stretch(first=first, count=count, contrast_db=contrast)
        held = min(self.strength * self.max_cut_db, self.gentle_cut_db)
        if partial:
            stretch.reliable = False
            stretch.note = (
                "the recording is shorter than one {:g} ms analysis frame, so the "
                "noise profile is unreliable and the cut was held to {:g} dB".format(
                    self.frame_ms, held
                )
            )
        elif seconds < MIN_PROFILE_SECONDS:
            stretch.reliable = False
            stretch.note = (
                "the recording is too short to hold a quiet stretch of {:g} s to "
                "learn the noise from, so the noise profile is unreliable and the "
                "cut was held to {:g} dB".format(MIN_PROFILE_SECONDS, held)
            )
        elif contrast < self.min_contrast_db:
            stretch.reliable = False
            stretch.note = (
                "no quiet stretch: the quietest part ({:.2f}-{:.2f} s) is only {:.1f} "
                "dB below the loudest, so it is probably not a pause and the noise "
                "profile is unreliable. The cut was held to {:g} dB; pass "
                "noise_profile= learned from a noise-only clip to clean "
                "harder".format(start_s, end_s, contrast, held)
            )
        else:
            stretch.note = (
                "the quietest stretch sits {:.1f} dB below the loudest parts, so it "
                "reads as a pause".format(contrast)
            )
        return stretch

    def _has_digital_pause(self, stft: Stft, levels: np.ndarray) -> bool:
        """True when a run of full frames at least MIN_PROFILE_SECONDS long is digital silence."""
        zero = stft.full_frames() & (levels <= SILENT_POWER)
        need = max(1, int(math.ceil(MIN_PROFILE_SECONDS * stft.sample_rate / stft.hop)))
        return _longest_run(zero) >= need

    def _floor(self, frequencies: np.ndarray, cut_db: float) -> Tuple[np.ndarray, float]:
        """Per-bin minimum gain, and the deepest cut allowed in the speech band."""
        depth = np.full(frequencies.shape, cut_db)
        speech_cut = cut_db
        if self.preserve_speech:
            speech_cut = min(cut_db, self.speech_cut_db)
            low, high = self.speech_band
            taper = 1.0 / 3.0  # octaves over which the protection fades out
            weight = np.zeros(frequencies.shape)
            weight[(frequencies >= low) & (frequencies <= high)] = 1.0
            with np.errstate(divide="ignore"):
                octaves_below = np.log2(np.maximum(frequencies, 1e-9) / low)
                octaves_above = np.log2(np.maximum(frequencies, 1e-9) / high)
            below = (octaves_below < 0.0) & (octaves_below > -taper)
            weight[below] = 0.5 + 0.5 * np.cos(np.pi * octaves_below[below] / taper)
            above = (octaves_above > 0.0) & (octaves_above < taper)
            weight[above] = 0.5 + 0.5 * np.cos(np.pi * octaves_above[above] / taper)
            depth = cut_db * (1.0 - weight) + speech_cut * weight
        return 10.0 ** (-depth / 20.0), speech_cut

    def _silent_report(self, channel: int) -> ChannelReport:
        """The report for a channel with nothing in it."""
        return ChannelReport(
            channel=channel,
            silent=True,
            profile_source="none",
            quiet_stretch=None,
            profile_reliable=False,
            reliability_note="the channel is digital silence",
            contrast_db=None,
            noise_floor_dbfs=None,
            max_cut_db=0.0,
            speech_cut_db=0.0,
            noise_reduction_db=0.0,
            snr_before=None,
            snr_after=None,
        )

    def _clean_channel(
        self,
        samples: np.ndarray,
        stft: Stft,
        given: Optional[np.ndarray],
        channel: int,
    ) -> Tuple[np.ndarray, ChannelReport, np.ndarray, np.ndarray]:
        """Gate one channel. Returns samples, report, profile and power sums."""
        padded = stft.pad(samples)
        levels = self._levels(stft, padded)
        sums = np.zeros(4)
        if not np.any(levels > SILENT_POWER):
            # Digital silence: nothing to learn, nothing to remove, and above
            # all nothing to amplify. Hand it back exactly.
            return samples.copy(), self._silent_report(channel), np.zeros(stft.n_bins), sums

        quiet = None  # type: Optional[Tuple[float, float]]
        if given is not None:
            profile = given
            source = "given"
            reliable = True
            note = "the noise profile was given by the caller"
            contrast = None
        else:
            stretch = self._find_stretch(stft, levels)
            spectra = stft.analyse(padded, stretch.first, stretch.first + stretch.count)
            power = stft.power(spectra)
            profile = np.median(power, axis=0) / _median_bias(stretch.count)
            source = "quietest stretch"
            reliable = stretch.reliable
            note = stretch.note
            contrast = stretch.contrast_db
            quiet = stft.frame_seconds(stretch.first, stretch.first + stretch.count - 1)
            if not reliable and self._has_digital_pause(stft, levels):
                # The only real pause is exact digital zeros: the recording has
                # no noise floor to learn, and the unreliable stretch found
                # instead sits inside the signal. Cutting there would invent
                # noise and damage a clean recording, so hand it back as is.
                report = self._silent_report(channel)
                report.silent = False
                report.profile_source = "digital silence"
                report.profile_reliable = True
                report.reliability_note = (
                    "the only pauses are exact digital silence, so there is no "
                    "noise floor to remove; the channel was left unchanged"
                )
                return samples.copy(), report, np.zeros(stft.n_bins), sums

        cut_db = self.strength * self.max_cut_db
        if not reliable:
            cut_db = min(cut_db, self.gentle_cut_db)
        floor, speech_cut = self._floor(stft.frequencies, cut_db)

        hop_ms = 1000.0 * stft.hop / stft.sample_rate
        bin_hz = stft.sample_rate / float(stft.n_fft)
        decide_t, decide_f = self._decide_frames, self._decide_bins
        smooth_t = int(round(self.time_smoothing_ms / hop_ms))
        smooth_f = int(round(self.freq_smoothing_hz / bin_hz))
        context = decide_t + smooth_t
        span = self.pass_db - self.threshold_db
        reference = profile + TINY

        accumulator = stft.new_output()
        gain_power = np.zeros(stft.n_bins)
        power_in = 0.0
        power_out = 0.0
        for first in range(0, stft.n_frames, self.block_frames):
            stop = min(stft.n_frames, first + self.block_frames)
            lead = max(0, first - context)
            tail = min(stft.n_frames, stop + context)
            spectra = stft.analyse(padded, lead, tail)
            power = stft.power(spectra)
            # Compare each bin with its own profile first, then smooth the
            # ratio. Smoothing power across frequency before comparing would let
            # a loud bin (a hum line) spill into a quiet neighbour and read there
            # as signal far above that neighbour's own tiny noise floor.
            ratio = (power + TINY) / reference
            decided = smooth(smooth(ratio, decide_t, axis=0), decide_f, axis=1)
            above = 10.0 * np.log10(decided)
            ramp = np.clip((above - self.threshold_db) / span, 0.0, 1.0)
            keep = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep: no hard edge
            keep = smooth(smooth(keep, smooth_t, axis=0), smooth_f, axis=1)
            gain = floor + (1.0 - floor) * keep
            inner = slice(first - lead, stop - lead)
            gain = gain[inner]
            power = power[inner]
            stft.overlap_add(accumulator, spectra[inner] * gain, first)
            squared = gain * gain
            gain_power += squared.sum(axis=0)
            power_in += float(power.sum())
            power_out += float((squared * power).sum())

        cleaned = stft.finish(accumulator)
        frames = float(stft.n_frames)
        noise_in = float(profile.sum())
        noise_out = float((profile * gain_power / frames).sum())
        signal_in = max(power_in / frames - noise_in, 0.0)
        signal_out = max(power_out / frames - noise_out, 0.0)
        sums[:] = (noise_in, noise_out, signal_in, signal_out)
        reduction = _to_db(noise_in) - _to_db(noise_out) if noise_in > 0.0 else 0.0
        report = ChannelReport(
            channel=channel,
            silent=False,
            profile_source=source,
            quiet_stretch=quiet,
            profile_reliable=reliable,
            reliability_note=note,
            contrast_db=contrast,
            noise_floor_dbfs=_to_db(noise_in) if noise_in > 0.0 else None,
            max_cut_db=float(cut_db),
            speech_cut_db=float(speech_cut),
            noise_reduction_db=float(max(reduction, 0.0)),
            snr_before=_snr(signal_in, noise_in),
            snr_after=_snr(signal_out, noise_out),
            noise_bands_dbfs=octave_bands(profile, stft.frequencies),
        )
        return cleaned, report, profile.copy(), sums

    def _fit_inside_rails(self, output: np.ndarray, loaded: Loaded) -> Tuple[np.ndarray, float]:
        """Turn the whole output down, never clip it, if it would pass full scale.

        The rails are the input format's own: an 8-bit WAV stops at +127/128, so
        the positive side is checked against that and not against 1.0.
        """
        if output.size == 0:
            return output, 0.0
        gain = 1.0
        top = float(np.max(output))
        bottom = float(np.min(output))
        if top > loaded.positive_rail:
            gain = min(gain, HEADROOM * loaded.positive_rail / top)
        if bottom < loaded.negative_rail:
            gain = min(gain, HEADROOM * loaded.negative_rail / bottom)
        if gain >= 1.0:
            return output, 0.0
        return output * gain, 20.0 * math.log10(gain)


def _snr(signal: float, noise: float) -> Optional[float]:
    """Signal-to-noise ratio in dB, or ``None`` when either side is unmeasurable."""
    if noise <= 0.0 or signal <= 0.0:
        return None
    ratio = 10.0 * math.log10(signal / noise)
    return ratio if ratio >= LOWEST_SNR_DB else None


def clean(
    audio: Any,
    *,
    sample_rate: Optional[int] = None,
    strength: float = 0.8,
    noise_profile: Optional[np.ndarray] = None,
    preserve_speech: bool = True,
) -> CleanResult:
    """Remove steady background noise (hum, hiss, fans) from a speech recording.

    Spectral gating: the noise spectrum is learned from the quietest stretch of
    the recording, or taken from ``noise_profile``, and every time-frequency bin
    is turned down by how close it sits to that profile, with smoothing across
    time and frequency so the result does not warble. A heuristic, not a model.

    Args:
        audio: A ``.wav`` path, a numpy array, or ``(samples, sample_rate)``.
            The caller's array is never modified.
        sample_rate: Required when ``audio`` is a bare array.
        strength: 0 to 1, how hard to cut. 0 leaves the audio untouched; the
            default 0.8 cuts noise-only bins by up to 24 dB.
        noise_profile: A profile from :func:`noise_profile`, or a previous
            result's ``.noise_profile``, used instead of learning one.
        preserve_speech: Keep a floor in the 300-3400 Hz band so speech is never
            gated away entirely.

    Returns:
        A :class:`CleanResult`; ``.audio`` has exactly the input's shape.
    """
    gate = SpectralGate(strength=strength, preserve_speech=preserve_speech)
    return gate.clean(audio, sample_rate=sample_rate, noise_profile=noise_profile)


def noise_profile(
    audio: Any, *, sample_rate: Optional[int] = None, frame_ms: float = 32.0
) -> np.ndarray:
    """Learn a noise profile from a clip that holds only the noise.

    Record a few seconds of the room, the fan or the hum with nobody talking,
    learn its profile here, and pass it to :func:`clean` as ``noise_profile=``
    to clean recordings that never pause long enough to learn it themselves.

    Args:
        audio: A ``.wav`` path, a numpy array, or ``(samples, sample_rate)``.
        sample_rate: Required when ``audio`` is a bare array. Must match the
            recordings the profile will be used on.
        frame_ms: Analysis frame length; leave at 32 unless the gate that will
            use the profile was built with another.

    Returns:
        Per-bin noise power: 1-D for a one-channel clip, ``(n_channels,
        n_bins)`` otherwise.
    """
    return SpectralGate(frame_ms=frame_ms).learn_profile(audio, sample_rate=sample_rate)
