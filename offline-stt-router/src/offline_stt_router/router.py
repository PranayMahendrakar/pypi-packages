"""The ``Router`` class: probe, choose, transcribe, with every knob exposed."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._audio import AudioInput
from ._engines import (
    Engine,
    Finder,
    find_whisper_cpp,
    probe_faster_whisper,
    probe_openai_whisper,
    probe_vosk,
    probe_whisper_cpp,
    run_custom_probe,
)
from ._machine import Machine
from ._models import (
    FASTER_WHISPER,
    OPENAI_WHISPER,
    VOSK,
    WHISPER_CPP,
    LocalModel,
    ModelScan,
    inspect_path,
    scan_models,
)
from ._plan import Choice, plan
from ._run import Attempt, ModelCache, Transcript, TranscriptionFailed, run_option

logger = logging.getLogger(__name__)

Probe = Callable[[], Any]
TranscribeFn = Callable[..., Any]

BUILTIN_ENGINES = (FASTER_WHISPER, OPENAI_WHISPER, WHISPER_CPP, VOSK)


class Router:
    """Finds local speech-to-text engines and picks one for this machine.

    ``machine`` - a fixed ``Machine`` to plan for (default: read this one on
    every call). ``model_dirs`` - extra folders to search for models of any
    kind. ``search_paths`` - folders to look for engine packages in instead of
    ``sys.path``. ``builtins=False`` leaves out the four built-in engines, so
    only engines you ``register()`` are considered.
    """

    def __init__(
        self,
        *,
        machine: Optional[Machine] = None,
        model_dirs: Sequence[str] = (),
        search_paths: Optional[Sequence[str]] = None,
        builtins: bool = True,
    ) -> None:
        if machine is not None and not isinstance(machine, Machine):
            raise TypeError("machine must be a Machine or None")
        self._machine = machine
        self.model_dirs: List[str] = [os.fspath(d) for d in model_dirs]
        self._finder = Finder(search_paths)
        self._builtins = bool(builtins)
        self._custom: "OrderedDict[str, Tuple[Probe, Optional[TranscribeFn]]]" = OrderedDict()
        self._lock = threading.Lock()
        self._cache = ModelCache()
        self._scan_notes: List[str] = []

    # ------------------------------------------------------------------ registry

    def register(self, name: str, probe: Probe, transcribe: Optional[TranscribeFn] = None) -> None:
        """Add a custom engine.

        ``probe()`` says whether it is installed: return ``True``/``False``, a
        dict of ``Engine`` fields (``version``, ``languages``, ``quality``,
        ``realtime_factor``, ``ram_gb``, ``models_on_disk``...) or an ``Engine``.
        ``transcribe(audio, *, language, model, model_path, device, **kw)``
        returns the text; without it the engine can be chosen but not run.
        Registering a built-in name replaces the built-in probe.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("an engine name must be a non-empty string")
        if not callable(probe):
            raise TypeError("probe must be callable, got {0}".format(type(probe).__name__))
        if transcribe is not None and not callable(transcribe):
            raise TypeError("transcribe must be callable or None, got {0}".format(type(transcribe).__name__))
        key = name.strip()
        if key.lower() in BUILTIN_ENGINES:
            logger.info("custom engine %r replaces the built-in probe", key)
        with self._lock:
            self._custom[key] = (probe, transcribe)

    def unregister(self, name: str) -> bool:
        """Remove a custom engine. Returns whether one was registered."""
        with self._lock:
            return self._custom.pop(str(name).strip(), None) is not None

    @property
    def registered(self) -> List[str]:
        """Names of the custom engines, in registration order."""
        return list(self._custom)

    # ------------------------------------------------------------------ probing

    def machine(self) -> Machine:
        """The machine being planned for."""
        return self._machine if self._machine is not None else Machine.detect()

    def available(self) -> List[Engine]:
        """Every engine this router knows, installed or not, with its state.

        Imports no engine, runs no binary, downloads nothing, and never raises
        because an engine is broken: that engine comes back with
        ``status="failing"`` and the reason in ``error``.
        """
        with self._lock:
            custom = list(self._custom.items())
        overridden = {name.lower() for name, _ in custom}
        self._finder.refresh()
        engines: List[Engine] = []
        self._scan_notes = []
        if self._builtins:
            binary, problem = find_whisper_cpp()
            try:
                scan = scan_models(self.model_dirs, binary)
            except Exception as exc:  # a filesystem that misbehaves
                logger.warning("model scan failed: %s", exc)
                scan = ModelScan()
                self._scan_notes.append("the model folders could not be read ({0})".format(exc))
            self._scan_notes.extend(scan.notes_for("any"))
            probes = (
                (FASTER_WHISPER, lambda: probe_faster_whisper(self._finder, scan)),
                (OPENAI_WHISPER, lambda: probe_openai_whisper(self._finder, scan)),
                (WHISPER_CPP, lambda: probe_whisper_cpp(self._finder, scan, binary, problem)),
                (VOSK, lambda: probe_vosk(self._finder, scan)),
            )
            for name, probe in probes:
                if name in overridden:
                    continue
                try:
                    engines.append(probe())
                except Exception as exc:
                    logger.warning("probing %s failed: %s", name, exc)
                    engines.append(Engine(name=name, installed=True,
                                          error="probing it failed: {0}".format(exc)).finish())
        for name, (probe, transcribe) in custom:
            engines.append(run_custom_probe(name, probe, transcribe is not None))
        return engines

    def engine(self, name: str) -> Engine:
        """One engine from ``available()``, by name. Raises ``KeyError`` if unknown."""
        for engine in self.available():
            if engine.name.lower() == str(name).strip().lower():
                return engine
        raise KeyError("no engine called {0!r}".format(name))

    # ------------------------------------------------------------------ choosing

    def choose(self, *, language: Optional[str] = "en", prefer: str = "balanced",
               max_ram_gb: Optional[float] = None) -> Choice:
        """Pick an engine, model and device. Always returns a ``Choice``.

        ``prefer`` is ``"fast"``, ``"balanced"`` or ``"accurate"``.
        ``max_ram_gb`` caps the RAM a model may use (it never raises the limit
        above what is free). ``language=None`` means auto-detect.
        """
        engines = self.available()
        choice = plan(engines, self.machine(), language=language, prefer=prefer, max_ram_gb=max_ram_gb)
        choice.notes.extend(self._scan_notes)
        return choice

    # ------------------------------------------------------------------ running

    def transcribe(self, audio: Any, **kw: Any) -> str:
        """Transcribe with the chosen engine. Only local models; never downloads.

        ``audio`` is a file path, the bytes of an audio file, or float samples
        (with ``sample_rate=``). Keywords: ``language``, ``prefer``,
        ``max_ram_gb`` as for ``choose()``; ``engine=`` and ``model=`` (a name
        or a path) to insist on one; ``fallback=False`` to stop after the first
        failure. Anything else goes to the engine. Raises
        ``TranscriptionFailed`` when nothing installed can do it.
        """
        return self.transcribe_detailed(audio, **kw).text

    def transcribe_detailed(
        self,
        audio: Any,
        *,
        language: Optional[str] = "en",
        prefer: str = "balanced",
        max_ram_gb: Optional[float] = None,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        fallback: bool = True,
        sample_rate: Optional[int] = None,
        **extra: Any,
    ) -> Transcript:
        """Like ``transcribe()`` but returns a ``Transcript`` saying which engine ran."""
        started = time.perf_counter()
        with AudioInput.from_user(audio, sample_rate) as source:
            engines = self.available()
            machine = self.machine()
            extra_models: List[LocalModel] = []
            if model is not None:
                model = os.fspath(model)
                if os.path.exists(model):
                    found, note = inspect_path(model)
                    if found is None:
                        raise ValueError("{0} is not a model any engine here can load{1}".format(
                            model, ": " + note if note else ""))
                    extra_models.append(found)
                    engine = engine or found.engine
            choice = plan(engines, machine, language=language, prefer=prefer, max_ram_gb=max_ram_gb,
                          runnable_only=True, only_engine=engine, only_model=model,
                          extra_models=extra_models)
            if not choice.ok or choice.option is None:
                hint = ""
                if choice.missing:
                    hint = " To fix: " + "; ".join(choice.missing[:2])
                raise TranscriptionFailed("cannot transcribe: " + choice.reason + hint, choice=choice)
            by_name: Dict[str, Engine] = {e.name: e for e in engines}
            options = [choice.option] + (list(choice.alternatives) if fallback else [])
            attempts: List[Attempt] = []
            for option in options:
                target = by_name[option.engine]
                custom_fn = self._custom.get(option.engine, (None, None))[1] if target.kind == "custom" else None
                begun = time.perf_counter()
                try:
                    text = run_option(option, target, source, choice.language, extra, machine,
                                      self._cache, custom_fn)
                except Exception as exc:
                    message = "{0}: {1}".format(type(exc).__name__, exc)[:500]
                    attempts.append(Attempt(option.engine, option.model, option.device, False, message,
                                            time.perf_counter() - begun))
                    logger.warning("%s failed (%s)%s", option.label(), message,
                                   "; trying the next option" if option is not options[-1] else "")
                    continue
                attempts.append(Attempt(option.engine, option.model, option.device, True, None,
                                        time.perf_counter() - begun))
                return Transcript(text=text, engine=option.engine, model=option.model,
                                  device=option.device, language=choice.language,
                                  seconds=time.perf_counter() - started, attempts=attempts)
            lines = "; ".join("{0} {1}: {2}".format(a.engine, a.model or "", a.error) for a in attempts)
            raise TranscriptionFailed("every option failed - " + lines, choice=choice, attempts=attempts)

    def unload(self) -> None:
        """Forget the model kept loaded between calls, freeing its memory."""
        self._cache.clear()
