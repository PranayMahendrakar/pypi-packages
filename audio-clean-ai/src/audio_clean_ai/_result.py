"""The result objects: what was done to the audio, and how sure the estimate is."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._audio import from_layout, write_wav

OCTAVE_CENTRES = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
"""Nominal octave band centres used to say where the noise lives."""


def _round(value: Optional[float], places: int = 2) -> Optional[float]:
    """Round a float for reporting, passing ``None`` and non-finite through as None."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, places)


def _fmt_db(value: Optional[float]) -> str:
    """``12.3 dB`` or ``n/a``."""
    return "n/a" if value is None else "{:.1f} dB".format(value)


def octave_bands(profile: np.ndarray, frequencies: np.ndarray) -> Dict[str, float]:
    """Noise power per octave band, in dB relative to full scale.

    Args:
        profile: One channel's noise profile (per-bin power).
        frequencies: The centre frequency of each bin, in Hz.

    Returns:
        ``{"125 Hz": -48.2, ...}`` for every band that holds at least one bin
        and some noise. Wider bands hold more bins, so white noise (hiss) reads
        loudest at the top, exactly as it sounds.
    """
    bands = {}  # type: Dict[str, float]
    for centre in OCTAVE_CENTRES:
        low, high = centre / math.sqrt(2.0), centre * math.sqrt(2.0)
        inside = (frequencies >= low) & (frequencies < high)
        if not np.any(inside):
            continue
        power = float(np.sum(profile[inside]))
        if power <= 0.0:
            continue
        label = "{:g} Hz".format(centre)
        bands[label] = round(10.0 * math.log10(power), 1)
    return bands


@dataclass
class ChannelReport:
    """What happened to one channel.

    Attributes:
        channel: Zero-based channel index.
        silent: True when the channel is digital silence and was left alone.
        profile_source: ``"quietest stretch"``, ``"given"`` or ``"none"``.
        quiet_stretch: ``(start_s, end_s)`` the profile was learned from, or
            ``None`` when it was given or there was nothing to learn.
        profile_reliable: False when the profile could not be trusted, in which
            case cleaning was held to a gentle cut.
        reliability_note: Why the profile was or was not trusted.
        contrast_db: How far the loudest parts sit above the quiet stretch.
        noise_floor_dbfs: Level of the learned noise, dB relative to full scale.
        max_cut_db: The deepest cut applied to a noise-only bin in this channel.
        speech_cut_db: The deepest cut allowed inside the 300-3400 Hz speech
            band (equal to ``max_cut_db`` when speech is not being preserved).
        noise_reduction_db: Estimated drop in the noise's level.
        snr_before: Estimated signal-to-noise ratio of the input, in dB.
        snr_after: Estimated signal-to-noise ratio of the output, in dB.
        noise_bands_dbfs: Noise level per octave band.
    """

    channel: int
    silent: bool
    profile_source: str
    quiet_stretch: Optional[Tuple[float, float]]
    profile_reliable: bool
    reliability_note: str
    contrast_db: Optional[float]
    noise_floor_dbfs: Optional[float]
    max_cut_db: float
    speech_cut_db: float
    noise_reduction_db: float
    snr_before: Optional[float]
    snr_after: Optional[float]
    noise_bands_dbfs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """The report as a JSON-safe dict."""
        stretch = None
        if self.quiet_stretch is not None:
            stretch = {
                "start_s": round(float(self.quiet_stretch[0]), 3),
                "end_s": round(float(self.quiet_stretch[1]), 3),
            }
        return {
            "channel": int(self.channel),
            "silent": bool(self.silent),
            "profile_source": self.profile_source,
            "quiet_stretch": stretch,
            "profile_reliable": bool(self.profile_reliable),
            "reliability_note": self.reliability_note,
            "contrast_db": _round(self.contrast_db),
            "noise_floor_dbfs": _round(self.noise_floor_dbfs),
            "max_cut_db": _round(self.max_cut_db),
            "speech_cut_db": _round(self.speech_cut_db),
            "noise_reduction_db": _round(self.noise_reduction_db),
            "snr_before": _round(self.snr_before),
            "snr_after": _round(self.snr_after),
            "noise_bands_dbfs": dict(self.noise_bands_dbfs),
        }

    def describe(self) -> str:
        """One or two lines on where this channel's profile came from."""
        if self.silent:
            return "digital silence, left unchanged"
        if self.profile_source == "given":
            where = "noise profile given by the caller"
        elif self.quiet_stretch is not None:
            where = "noise learned from the quietest stretch {:.2f}-{:.2f} s".format(
                self.quiet_stretch[0], self.quiet_stretch[1]
            )
        else:
            where = "no noise profile"
        floor = (
            "noise floor n/a"
            if self.noise_floor_dbfs is None
            else "noise floor {:.1f} dBFS".format(self.noise_floor_dbfs)
        )
        trust = "reliable" if self.profile_reliable else "UNRELIABLE, cut held back"
        text = "{}, {} ({})".format(floor, where, trust)
        if self.noise_bands_dbfs:
            loudest = max(self.noise_bands_dbfs.items(), key=lambda item: item[1])[0]
            text += "; noise is strongest in the {} octave".format(loudest)
        return text


@dataclass
class CleanResult:
    """The cleaned audio and an account of what was removed.

    Attributes:
        audio: The cleaned samples, float64 in -1 to 1, in exactly the shape and
            layout of the input (1-D, samples x channels, or channels x samples).
        sample_rate: Frames per second.
        noise_reduction_db: Estimated drop in the noise's level, averaged over
            the recording and all channels. An estimate from the noise profile,
            not a measurement against a clean reference (there is none).
        noise_profile: The per-bin noise power the gate worked from: 1-D for a
            one-channel recording, ``(n_channels, n_bins)`` otherwise. Its sum
            is the noise's mean square, so ``10*log10(profile.sum())`` is the
            noise level in dB relative to full scale. Hand it back in as
            ``noise_profile=`` to clean another recording from the same room.
        snr_before: Estimated signal-to-noise ratio of the input in dB, or
            ``None`` when no signal stands clearly above the noise (below about
            -10 dB the estimate would be error, not measurement).
        snr_after: The same estimate for the output.
        profile_reliable: False when any channel's profile could not be trusted.
        strength: The strength the gate ran at, 0 to 1.
        preserve_speech: Whether the 300-3400 Hz floor was kept.
        frequencies: Centre frequency in Hz of each profile bin.
        channels: One :class:`ChannelReport` per channel.
        source: The input's name (file name, or ``<array>``).
        n_samples: Samples per channel, identical for input and output.
        n_channels: Channel count.
        channel_axis: ``None`` for 1-D audio, ``1`` for samples x channels,
            ``0`` for channels x samples.
        output_gain_db: 0, or the negative gain applied so the output stays
            inside full scale without clipping.
        clipping_before: Samples at the positive and negative rails of the input.
        clipping_after: The same count for the output.
        positive_rail: The input format's positive full-scale value (127/128 for
            8-bit WAV, 1.0 for float).
        negative_rail: The input format's negative full-scale value.
        source_bits: PCM bit depth of the input file, if it was one.
        warnings: Things that went against the grain and deserve a look.
        notes: Decisions taken on the caller's behalf.
    """

    audio: np.ndarray
    sample_rate: int
    noise_reduction_db: float
    noise_profile: np.ndarray
    snr_before: Optional[float]
    snr_after: Optional[float]
    profile_reliable: bool
    strength: float
    preserve_speech: bool
    frequencies: np.ndarray
    channels: List[ChannelReport]
    source: str
    n_samples: int
    n_channels: int
    channel_axis: Optional[int]
    output_gain_db: float
    clipping_before: Dict[str, int]
    clipping_after: Dict[str, int]
    positive_rail: float
    negative_rail: float
    source_bits: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        """Length of the recording in seconds."""
        return self.n_samples / float(self.sample_rate) if self.sample_rate else 0.0

    @property
    def snr_improvement_db(self) -> Optional[float]:
        """``snr_after - snr_before``, or ``None`` if either is unknown."""
        if self.snr_before is None or self.snr_after is None:
            return None
        return self.snr_after - self.snr_before

    def summary(self) -> str:
        """A short human-readable account, plain ASCII apart from the file name."""
        layout = "mono" if self.n_channels == 1 else "{} channels".format(self.n_channels)
        where = "[{}, {:.3f} s at {} Hz, {}]".format(
            self.source, self.duration_s, self.sample_rate, layout
        )
        live = [report for report in self.channels if not report.silent]
        if self.n_samples == 0:
            head = "audio clean: nothing to do, the recording is empty"
        elif not live:
            head = "audio clean: nothing to do, the recording is digital silence"
        else:
            head = "audio clean: noise down {} (estimated), SNR {} before, {} after".format(
                _fmt_db(self.noise_reduction_db),
                _fmt_db(self.snr_before),
                _fmt_db(self.snr_after),
            )
        lines = [head + "  " + where]
        if live:
            if self.n_channels == 1:
                lines.append("  " + self.channels[0].describe())
            else:
                for report in self.channels:
                    lines.append(
                        "  channel {}: {}".format(report.channel + 1, report.describe())
                    )
            deepest = max(report.max_cut_db for report in live)
            setting = "  strength {:.2f}: noise-only bins cut by up to {:.0f} dB".format(
                self.strength, deepest
            )
            if self.preserve_speech:
                speech = max(report.speech_cut_db for report in live)
                setting += ", the 300-3400 Hz speech band by at most {:.0f} dB".format(
                    speech
                )
            lines.append(setting)
        for warning in self.warnings:
            lines.append("  warning: " + warning)
        for note in self.notes:
            lines.append("  note: " + note)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Everything except the samples themselves, as a JSON-safe dict."""
        return {
            "source": self.source,
            "sample_rate": int(self.sample_rate),
            "n_samples": int(self.n_samples),
            "n_channels": int(self.n_channels),
            "duration_s": round(self.duration_s, 6),
            "strength": round(float(self.strength), 4),
            "preserve_speech": bool(self.preserve_speech),
            "noise_reduction_db": _round(self.noise_reduction_db),
            "snr_before": _round(self.snr_before),
            "snr_after": _round(self.snr_after),
            "snr_improvement_db": _round(self.snr_improvement_db),
            "profile_reliable": bool(self.profile_reliable),
            "noise_profile_bins": int(self.frequencies.size),
            "output_gain_db": _round(self.output_gain_db, 3),
            "clipping": {
                "positive_rail": float(self.positive_rail),
                "negative_rail": float(self.negative_rail),
                "before": {k: int(v) for k, v in self.clipping_before.items()},
                "after": {k: int(v) for k, v in self.clipping_after.items()},
            },
            "channels": [report.to_dict() for report in self.channels],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    def save(self, path: Any, bits: Optional[int] = None) -> str:
        """Write the cleaned audio to a PCM ``.wav`` file with the :mod:`wave` module.

        Args:
            path: Destination file.
            bits: 8, 16, 24 or 32. ``None`` keeps the input file's PCM depth, or
                16-bit when the input was an array or a float WAV.

        Returns:
            The path written.
        """
        depth = bits if bits is not None else (self.source_bits or 16)
        frames = from_layout(self.audio, self.channel_axis)
        return write_wav(path, frames, self.sample_rate, bits=depth)
