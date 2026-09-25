"""Running the chosen engine. The only module that imports an engine, and only here.

Every model is loaded from a local path: faster-whisper gets a folder,
openai-whisper a ``.pt`` file, Vosk a model folder, whisper.cpp a ggml file.
None of these can trigger a download, because a download only happens when an
engine is given a model *name* instead of a path, and this module never does.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ._audio import TARGET_RATE, AudioInput
from ._engines import Engine
from ._machine import Machine
from ._models import FASTER_WHISPER, OPENAI_WHISPER, VOSK, WHISPER_CPP
from ._plan import Choice, Option

logger = logging.getLogger(__name__)

TranscribeFn = Callable[..., Any]


class TranscriptionFailed(RuntimeError):
    """Nothing could transcribe the audio. ``.choice`` and ``.attempts`` say why."""

    def __init__(self, message: str, choice: Optional[Choice] = None, attempts: Optional[List["Attempt"]] = None) -> None:
        super().__init__(message)
        self.choice = choice
        self.attempts = list(attempts or [])


@dataclass
class Attempt:
    """One try at transcribing with one option."""

    engine: str
    model: Optional[str]
    device: str
    ok: bool
    error: Optional[str] = None
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "device": self.device,
            "ok": self.ok,
            "error": self.error,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class Transcript:
    """The text, and which engine produced it."""

    text: str
    engine: str
    model: Optional[str]
    device: str
    language: Optional[str]
    seconds: float
    attempts: List[Attempt] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "engine": self.engine,
            "model": self.model,
            "device": self.device,
            "language": self.language,
            "seconds": round(self.seconds, 3),
            "attempts": [a.to_dict() for a in self.attempts],
        }


class ModelCache:
    """Keeps the most recently used model loaded, so repeated calls are fast."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: Optional[tuple] = None
        self._value: Any = None

    def get(self, key: tuple, factory: Callable[[], Any]) -> Any:
        with self._lock:
            if self._key == key and self._value is not None:
                return self._value
            self._key, self._value = None, None
            value = factory()
            self._key, self._value = key, value
            return value

    def clear(self) -> None:
        with self._lock:
            self._key, self._value = None, None

    @property
    def loaded(self) -> Optional[tuple]:
        return self._key


def _join(parts: Any) -> str:
    return " ".join(str(p).strip() for p in parts if str(p).strip())


def _numpy_samples(audio: AudioInput) -> Any:
    import numpy as np

    return np.frombuffer(audio.samples_16k().tobytes(), dtype=np.float32).copy()


def _run_faster_whisper(option: Option, audio: AudioInput, language: Optional[str],
                        extra: Dict[str, Any], machine: Machine, cache: ModelCache) -> str:
    path = option.model_path
    if not path or not os.path.isdir(path):
        raise RuntimeError(
            "faster-whisper needs a model folder on disk and {0!r} is not one; this "
            "package never lets it download".format(path)
        )
    from faster_whisper import WhisperModel

    source = audio.path if audio.path else _numpy_samples(audio)

    def run(device: str, compute: str) -> str:
        key = (FASTER_WHISPER, path, device, compute, machine.threads)
        model = cache.get(key, lambda: WhisperModel(path, device=device, compute_type=compute,
                                                     cpu_threads=machine.threads))
        segments, _info = model.transcribe(source, language=language, **extra)
        return _join(segment.text for segment in segments)

    device = "cuda" if option.device == "cuda" else "cpu"
    compute = option.compute_type or ("float16" if device == "cuda" else "int8")
    try:
        return run(device, compute)
    except Exception as exc:
        if device != "cuda":
            raise
        logger.warning("faster-whisper on CUDA failed (%s); retrying on the CPU at int8", exc)
        cache.clear()
        return run("cpu", "int8")


def _run_openai_whisper(option: Option, audio: AudioInput, language: Optional[str],
                        extra: Dict[str, Any], machine: Machine, cache: ModelCache) -> str:
    path = option.model_path
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "openai-whisper needs a .pt checkpoint on disk and {0!r} is not one; this "
            "package never lets it download".format(path)
        )
    import whisper

    device = "cuda" if option.device == "cuda" else "cpu"
    if audio.decodable or audio.path is None:
        source: Any = _numpy_samples(audio)
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "openai-whisper needs ffmpeg on PATH to read {0} files; convert the audio "
                "to WAV or install ffmpeg".format((audio.format or "this").upper())
            )
        source = audio.path
    model = cache.get((OPENAI_WHISPER, path, device), lambda: whisper.load_model(path, device=device))
    options = dict(extra)
    options.setdefault("fp16", device == "cuda")
    result = model.transcribe(source, language=language, **options)
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(getattr(result, "text", result)).strip()


def _run_whisper_cpp(option: Option, audio: AudioInput, language: Optional[str], extra: Dict[str, Any],
                     machine: Machine, engine: Engine) -> str:
    binary = engine.location
    if not binary:
        raise RuntimeError("the whisper.cpp binary was not found")
    path = option.model_path
    if not path or not os.path.isfile(path):
        raise RuntimeError("whisper.cpp needs a ggml model file and {0!r} is not one".format(path))
    extra = dict(extra)
    timeout = extra.pop("timeout", None)
    more_args = [str(a) for a in extra.pop("args", [])]
    if extra:
        logger.warning("whisper.cpp ignores these options: %s", ", ".join(sorted(extra)))
    wav = audio.wav16k_path() if (audio.decodable or audio.path is None) else audio.path
    with tempfile.TemporaryDirectory(prefix="offline-stt-router-") as work:
        out_base = os.path.join(work, "transcript")
        command = [binary, "-m", path, "-f", wav, "-l", language or "auto",
                   "-t", str(machine.threads), "-nt", "-otxt", "-of", out_base] + more_args
        done = subprocess.run(command, capture_output=True, timeout=timeout)
        if done.returncode != 0:
            lines = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError("whisper.cpp exited with code {0}: {1}".format(
                done.returncode, lines[-1] if lines else "no error text"))
        text_file = out_base + ".txt"
        if os.path.isfile(text_file):
            with open(text_file, "r", encoding="utf-8", errors="replace") as handle:
                return _join(handle.read().splitlines())
        return _join(done.stdout.decode("utf-8", "replace").splitlines())


def _run_vosk(option: Option, audio: AudioInput, language: Optional[str], extra: Dict[str, Any],
              machine: Machine, cache: ModelCache) -> str:
    path = option.model_path
    if not path or not os.path.isdir(path):
        raise RuntimeError(
            "Vosk needs a model folder on disk and {0!r} is not one; this package never "
            "lets it download".format(path)
        )
    if extra:
        logger.warning("vosk ignores these options: %s", ", ".join(sorted(extra)))
    import vosk

    try:
        vosk.SetLogLevel(-1)
    except Exception:  # older builds without it
        pass
    model = cache.get((VOSK, path), lambda: vosk.Model(path))
    recognizer = vosk.KaldiRecognizer(model, TARGET_RATE)
    pcm = audio.pcm16_16k()
    pieces: List[str] = []
    step = 8000  # 4000 frames of 16-bit audio, a quarter of a second
    for start in range(0, len(pcm), step):
        if recognizer.AcceptWaveform(pcm[start:start + step]):
            pieces.append(json.loads(recognizer.Result()).get("text", ""))
    pieces.append(json.loads(recognizer.FinalResult()).get("text", ""))
    return _join(pieces)


def _call_custom(fn: TranscribeFn, audio: AudioInput, option: Option, language: Optional[str],
                 extra: Dict[str, Any]) -> str:
    kwargs: Dict[str, Any] = {
        "language": language,
        "model": option.model,
        "model_path": option.model_path,
        "device": option.device,
        "sample_rate": audio._rate if audio.format == "samples" else None,
    }
    kwargs.update(extra)
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        parameters = None
    if parameters is not None and not any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in parameters}
    result = fn(audio.original, **kwargs)
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"].strip()
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text.strip()
    raise TypeError("the transcribe function returned {0}; expected a string".format(type(result).__name__))


def run_option(option: Option, engine: Engine, audio: AudioInput, language: Optional[str],
               extra: Dict[str, Any], machine: Machine, cache: ModelCache,
               custom_fn: Optional[TranscribeFn]) -> str:
    """Transcribe with one option. Raises whatever the engine raises."""
    if custom_fn is not None:
        return _call_custom(custom_fn, audio, option, language, extra)
    if option.engine == FASTER_WHISPER:
        return _run_faster_whisper(option, audio, language, extra, machine, cache)
    if option.engine == OPENAI_WHISPER:
        return _run_openai_whisper(option, audio, language, extra, machine, cache)
    if option.engine == WHISPER_CPP:
        return _run_whisper_cpp(option, audio, language, extra, machine, engine)
    if option.engine == VOSK:
        return _run_vosk(option, audio, language, extra, machine, cache)
    raise RuntimeError("{0} has no transcribe function registered".format(option.engine))
