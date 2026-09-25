"""speech-quality: is this voice recording clean enough to transcribe or publish?

    from speech_quality import assess
    report = assess("interview.wav")
    print(report.summary())

Seven measures - level, clipping, noise, silence, speech, dynamics and
bandwidth - each give a raw number, a 0-100 score and a sentence you can act
on. WAV files are read with the standard library, so the only dependency is
numpy and nothing is ever downloaded.

These are signal measurements, not a perceptual model: the package tells you
that a recording clips, hisses or stops at 3.4 kHz, not what a listener would
score it out of five.
"""

from __future__ import annotations

from ._frames import Frames, Spectra, frame_signal, take_spectra
from .audio import Audio, load_audio, read_wav
from .core import assess, assess_batch, estimate_noise_floor, signal_to_noise
from .report import AudioReport, BatchReport, Metric
from .thresholds import DECISIVE_MEASURES, DEFAULT_THRESHOLDS, MEASURES, Thresholds

__version__ = "0.1.0"

__all__ = [
    "assess",
    "assess_batch",
    "signal_to_noise",
    "estimate_noise_floor",
    "AudioReport",
    "BatchReport",
    "Metric",
    "Audio",
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "MEASURES",
    "DECISIVE_MEASURES",
    "Frames",
    "Spectra",
    "load_audio",
    "read_wav",
    "frame_signal",
    "take_spectra",
    "__version__",
]
