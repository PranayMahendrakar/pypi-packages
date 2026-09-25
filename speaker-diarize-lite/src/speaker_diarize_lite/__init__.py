"""speaker-diarize-lite: work out who spoke when in a recording.

The default method is a lightweight heuristic built from numpy alone: an energy
detector finds speech, short windows are described by their log mel-band
energies and deltas, and the windows are clustered. It separates clearly
different voices and struggles with similar ones. Pass ``embed=`` to let a real
speaker-embedding model describe the windows instead.

    >>> from speaker_diarize_lite import diarize
    >>> result = diarize("meeting.wav")          # doctest: +SKIP
    >>> print(result.summary())                  # doctest: +SKIP
"""

from ._diarize import Diarizer, diarize
from ._result import Diarization, Segment
from ._wav import read_wav, write_wav

__version__ = "0.1.0"

__all__ = [
    "Diarization",
    "Diarizer",
    "Segment",
    "__version__",
    "diarize",
    "read_wav",
    "write_wav",
]
