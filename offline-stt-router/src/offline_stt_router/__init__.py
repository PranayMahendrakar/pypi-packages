"""offline-stt-router: find the speech-to-text engines installed on this machine
and pick the right one for its hardware and your language.

    import offline_stt_router as stt
    print(stt.choose(language="hi").summary())

The choice is a heuristic: engines and models are ranked with rough published
figures for memory, speed and accuracy, scaled to this machine's RAM, cores and
GPU. Probing imports no engine, runs no binary and never touches the network.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from ._engines import Engine
from ._languages import language_name, normalize_language
from ._machine import GPU, Machine
from ._models import LocalModel
from ._plan import PREFERENCES, Choice, Option
from ._run import Attempt, Transcript, TranscriptionFailed
from .router import Router

__version__ = "0.1.0"

__all__ = [
    "Attempt",
    "Choice",
    "Engine",
    "GPU",
    "LocalModel",
    "Machine",
    "Option",
    "PREFERENCES",
    "Router",
    "Transcript",
    "TranscriptionFailed",
    "__version__",
    "available",
    "choose",
    "language_name",
    "machine",
    "normalize_language",
    "register",
    "transcribe",
    "unregister",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())

_default = Router()


def available() -> List[Engine]:
    """Every speech-to-text engine this package knows, and whether it is installed.

    Built in: faster-whisper, openai-whisper, whisper.cpp (a binary on PATH) and
    vosk, plus anything added with ``register()``. Nothing is imported, run or
    downloaded; a broken install comes back as ``status="failing"``.
    """
    return _default.available()


def choose(*, language: Optional[str] = "en", prefer: str = "balanced",
           max_ram_gb: Optional[float] = None) -> Choice:
    """Pick the engine, model and device to use here. Never raises for "nothing installed".

    ``prefer`` is ``"fast"``, ``"balanced"`` or ``"accurate"``. ``max_ram_gb``
    caps the RAM a model may use. ``language=None`` asks for auto-detection.
    """
    return _default.choose(language=language, prefer=prefer, max_ram_gb=max_ram_gb)


def register(name: str, probe: Callable[[], Any], transcribe: Optional[Callable[..., Any]] = None) -> None:
    """Add a custom engine to the default router (see ``Router.register``)."""
    _default.register(name, probe, transcribe)


def unregister(name: str) -> bool:
    """Remove a custom engine from the default router."""
    return _default.unregister(name)


def transcribe(audio: Any, **kw: Any) -> str:
    """Transcribe ``audio`` with the engine ``choose()`` picks, from local models only.

    Raises ``TranscriptionFailed`` (a ``RuntimeError``) when nothing installed
    can do it; its message says what to install. Never downloads a model.
    """
    return _default.transcribe(audio, **kw)


def machine() -> Machine:
    """This computer's RAM, CPU cores and GPUs, as the router sees them."""
    return _default.machine()
