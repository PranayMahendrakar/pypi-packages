"""Every limit the assessment uses, in one place and overridable.

The defaults suit spoken-word recordings destined for transcription or for
publication as a podcast or voice-over. Pass a ``Thresholds`` or a dict of
overrides to any entry point to move them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional, Union

DB_FLOOR = -120.0
"""Decibel value standing in for digital zero, so nothing ever reports -inf."""

NEUTRAL_SCORE = 70.0
"""Score for a measure that genuinely could not be taken, neither good nor bad."""

GRADE_BANDS = (
    ("A", 85.0),
    ("B", 70.0),
    ("C", 55.0),
    ("D", 40.0),
    ("E", 25.0),
    ("F", 0.0),
)
"""Lower bound of each grade, best first."""

MEASURES = (
    "level",
    "clipping",
    "noise",
    "silence",
    "speech",
    "dynamics",
    "bandwidth",
)
"""The measures a report always carries, in reading order."""

MEASURE_WEIGHTS = {
    "level": 1.0,
    "clipping": 1.5,
    "noise": 1.5,
    "silence": 0.8,
    "speech": 1.2,
    "dynamics": 0.6,
    "bandwidth": 1.0,
}
"""How much each measure pulls on the overall score."""

DECISIVE_MEASURES = ("clipping", "noise", "speech", "bandwidth")
"""The measures whose failure settles the verdict on its own.

A weighted average cannot express "this one thing is fatal": no measure carries
enough of the total weight to drag a recording under the usable line by itself,
so a take that is perfect apart from being 7% clipped still averages out
respectably. These four are the faults nothing later puts right - clipped
samples are a destroyed waveform, noise under the voice is already mixed in, a
recording that does not behave like speech has nothing to transcribe, and a band
that stops at 3.4 kHz has lost its consonants. Level and silence describe faults
a later pass does fix, by raising the gain or trimming the dead air, and heavy
compression is a production choice rather than damage, so those three are left
to the average.
"""


@dataclass
class Thresholds:
    """The numbers that turn a measurement into a verdict.

    Attributes:
        target_rms_dbfs: the RMS level a healthy speech recording sits at.
        level_tolerance_db: how far from that target still counts as on target.
        clip_level: absolute sample value counted as at or near full scale.
        clip_ok_share: share of clipped samples still considered clean.
        noise_floor_percentile: which percentile of frame levels is the floor.
        signal_percentile: which percentile of frame levels stands for speech.
        min_snr_db: signal-to-noise below this is called out as an issue.
        stationary_spread_db: when the loud and quiet frames are within this
            many dB of each other the recording never goes quiet, so a noise
            floor cannot be separated from the signal at all.
        silence_drop_db: a frame this far under the loudest frame is silent.
        silence_floor_dbfs: absolute level under which a frame is silent
            whatever the rest of the recording does.
        max_edge_silence_s: leading or trailing silence longer than this is
            called out.
        max_silent_share: total silent share above this is called out.
        min_speech_share: estimated speech share below this is called out.
        speech_centroid_low_hz: lowest spectral centroid that can be speech.
        speech_centroid_high_hz: highest spectral centroid that can be speech.
        speech_band_low_hz: bottom of the band speech energy must fall in.
        speech_band_high_hz: top of that band.
        speech_band_share: how much of a frame's energy must lie in that band.
        min_modulation_db: syllable-rate level swing below which the recording
            is a steady tone or hum rather than speech.
        compressed_crest_db: crest factor under this reads as heavy compression.
        min_crest_db: crest factor under this is called out as an issue.
        bandwidth_floor_db: how far under the spectral peak still counts as
            real energy when looking for the top of the band.
        telephone_band_hz: bandwidth at or under this is telephone-band audio.
        min_bandwidth_hz: bandwidth under this is called out as an issue.
        upsampled_nyquist_share: bandwidth under this share of Nyquist means
            the recording was upsampled from a narrower original.
        frame_seconds: analysis frame length.
        hop_seconds: analysis frame step.
        max_spectral_frames: how many frames at most get a spectrum taken,
            spread evenly so the result stays deterministic.
        usable_score: overall score at or above which a recording is usable.
    """

    target_rms_dbfs: float = -20.0
    level_tolerance_db: float = 3.0

    clip_level: float = 0.995
    clip_ok_share: float = 0.0005

    noise_floor_percentile: float = 10.0
    signal_percentile: float = 90.0
    min_snr_db: float = 15.0
    stationary_spread_db: float = 3.0

    silence_drop_db: float = 35.0
    silence_floor_dbfs: float = -65.0
    max_edge_silence_s: float = 1.0
    max_silent_share: float = 0.6

    min_speech_share: float = 0.3
    speech_centroid_low_hz: float = 150.0
    speech_centroid_high_hz: float = 3500.0
    speech_band_low_hz: float = 80.0
    speech_band_high_hz: float = 4000.0
    speech_band_share: float = 0.55
    min_modulation_db: float = 2.0

    compressed_crest_db: float = 9.0
    min_crest_db: float = 6.0

    bandwidth_floor_db: float = 50.0
    telephone_band_hz: float = 4000.0
    min_bandwidth_hz: float = 3000.0
    upsampled_nyquist_share: float = 0.55

    frame_seconds: float = 0.032
    hop_seconds: float = 0.016
    max_spectral_frames: int = 600

    usable_score: float = 55.0

    def to_dict(self) -> Dict[str, Any]:
        """Every threshold as a JSON-safe mapping."""
        return asdict(self)

    def replace(self, **overrides: Any) -> "Thresholds":
        """A copy with some fields changed.

        Raises:
            ValueError: a name is not a threshold, or a value is not a number.
        """
        known = {field.name for field in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(
                "unknown threshold(s) {}; known names are {}".format(
                    ", ".join(repr(name) for name in unknown),
                    ", ".join(sorted(known)),
                )
            )
        merged = self.to_dict()
        for name, value in overrides.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "threshold {!r} wants a number, got {!r}".format(name, value)
                )
            merged[name] = value
        return Thresholds(**merged)


DEFAULT_THRESHOLDS = Thresholds()
"""The thresholds used when a call does not pass its own."""

ThresholdsLike = Union[Thresholds, Mapping[str, Any], None]


def resolve_thresholds(thresholds: ThresholdsLike = None) -> Thresholds:
    """Turn ``None``, a dict of overrides or a ``Thresholds`` into a ``Thresholds``.

    Args:
        thresholds: nothing, a mapping of names to override, or a ready object.

    Returns:
        The thresholds to measure against.

    Raises:
        ValueError: an override names something that is not a threshold.
        TypeError: the argument is not one of the accepted shapes.
    """
    if thresholds is None:
        return DEFAULT_THRESHOLDS
    if isinstance(thresholds, Thresholds):
        return thresholds
    if isinstance(thresholds, Mapping):
        return DEFAULT_THRESHOLDS.replace(**dict(thresholds))
    raise TypeError(
        "thresholds must be a Thresholds, a dict of overrides or None, got {}".format(
            type(thresholds).__name__
        )
    )


def grade_for(score: float) -> str:
    """The letter grade for an overall score, "A" (best) through "F"."""
    for letter, lower in GRADE_BANDS:
        if score >= lower:
            return letter
    return "F"


def optional_float(value: Optional[float]) -> Optional[float]:
    """Round an optional float for JSON, keeping ``None`` as ``None``."""
    if value is None:
        return None
    return round(float(value), 4)
