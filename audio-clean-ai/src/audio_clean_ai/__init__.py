"""audio-clean-ai: remove steady background noise from speech recordings, without a model.

    from audio_clean_ai import clean
    result = clean("interview.wav")
    result.save("interview-clean.wav")
    print(result.summary())

This is spectral gating, a heuristic signal-processing method, not a neural
denoiser. It learns what the noise sounds like from the quietest stretch of the
recording and turns down only that. It works on steady noise - hum, hiss, fans,
air conditioning - and does little for music, babble or a second speaker. It
needs nothing but numpy and downloads nothing.
"""

from __future__ import annotations

from ._result import ChannelReport, CleanResult
from .core import SpectralGate, clean, noise_profile

__version__ = "0.1.0"

__all__ = [
    "clean",
    "noise_profile",
    "SpectralGate",
    "CleanResult",
    "ChannelReport",
    "__version__",
]
