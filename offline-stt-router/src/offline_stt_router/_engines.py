"""Finding speech-to-text engines without importing them.

Every built-in probe works from ``importlib.util.find_spec``, package metadata
and files on disk. None of them imports faster-whisper, whisper, torch or vosk,
runs the whisper.cpp binary, or touches the network. An engine that is present
but cannot possibly import (a dependency missing, its native library gone) is
reported as installed-but-failing instead of raising.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._languages import normalize_language
from ._machine import detect_gpus, is_apple_silicon
from ._models import (
    FASTER_WHISPER,
    OPENAI_WHISPER,
    VOSK,
    WHISPER_CPP,
    LocalModel,
    ModelScan,
    whisper_cpp_dirs,
)

try:  # Python 3.9+
    import importlib.metadata as _metadata
except ImportError:  # pragma: no cover - 3.8 is not supported
    _metadata = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

READY = "ready"
NO_MODEL = "no-model"
FAILING = "failing"
NOT_INSTALLED = "not-installed"

#: Point this at a whisper.cpp binary with an unusual name (an old ``main``).
WHISPER_CPP_BIN_ENV = "WHISPER_CPP_BIN"
WHISPER_CPP_NAMES = ("whisper-cli", "whisper-cpp", "whisper.cpp", "whispercpp")

INSTALL_HINTS: Dict[str, str] = {
    FASTER_WHISPER: "pip install faster-whisper",
    OPENAI_WHISPER: "pip install openai-whisper (it brings PyTorch, a large download)",
    WHISPER_CPP: (
        "build whisper.cpp (https://github.com/ggml-org/whisper.cpp) and put "
        "whisper-cli on PATH or set WHISPER_CPP_BIN (macOS: brew install whisper-cpp)"
    ),
    VOSK: "pip install vosk",
}


@dataclass
class Engine:
    """One speech-to-text engine and what state it is in on this machine.

    ``status`` is ``"ready"``, ``"no-model"`` (installed, nothing to load),
    ``"failing"`` (installed but broken - see ``error``) or ``"not-installed"``.
    ``models_on_disk`` is filled even for an engine that is not installed, so
    you can see that installing it would put existing downloads to use.
    """

    name: str
    installed: bool = False
    version: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    needs_gpu: bool = False
    models_on_disk: List[LocalModel] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    status: str = NOT_INSTALLED
    error: Optional[str] = None
    can_use_gpu: bool = False
    location: Optional[str] = None
    kind: str = "builtin"
    install_hint: str = ""
    needs_model: bool = True
    runnable: bool = True
    quality: Optional[float] = None
    realtime_factor: Optional[float] = None
    ram_gb: Optional[float] = None
    vram_gb: Optional[float] = None

    @property
    def ready(self) -> bool:
        """Installed, working, and has what it needs to transcribe."""
        return self.status == READY

    @property
    def failing(self) -> bool:
        """Installed but broken."""
        return self.status == FAILING

    def finish(self) -> "Engine":
        """Work out ``status`` and ``languages`` from the other fields."""
        if self.needs_model:
            langs: List[str] = []
            for model in self.models_on_disk:
                for code in model.languages:
                    if code not in langs:
                        langs.append(code)
            self.languages = langs
        if not self.installed:
            self.status = NOT_INSTALLED
        elif self.error:
            self.status = FAILING
        elif self.needs_model and not self.models_on_disk:
            self.status = NO_MODEL
        else:
            self.status = READY
        return self

    def describe(self) -> str:
        """One line, plain ASCII."""
        head = self.name + (" " + self.version if self.version else "")
        if self.status == READY:
            state = "ready"
        elif self.status == FAILING:
            state = "installed but failing: " + (self.error or "unknown error")
        elif self.status == NO_MODEL:
            state = "installed, but no model on disk"
        else:
            state = "not installed"
        text = "{0} - {1}".format(head, state)
        if self.models_on_disk:
            names = ", ".join(m.name for m in self.models_on_disk[:6])
            more = len(self.models_on_disk) - 6
            if more > 0:
                names += " and {0} more".format(more)
            text += "; models on disk: " + names
        elif self.status == READY and not self.needs_model:
            text += "; needs no model file"
        return text

    def summary(self) -> str:
        """Human-readable text, plain ASCII."""
        lines = [self.describe()]
        if self.location:
            lines.append("  at: " + self.location)
        for model in self.models_on_disk:
            lines.append(
                "  model {0}: {1}, {2:.2f} GB{3}, {4}".format(
                    model.name,
                    model.family,
                    model.size_gb,
                    ", " + model.quantization if model.quantization else "",
                    model.describe_languages(),
                )
            )
        for note in self.notes:
            lines.append("  note: " + note)
        if self.status == NOT_INSTALLED and self.install_hint:
            lines.append("  install: " + self.install_hint)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "name": self.name,
            "installed": self.installed,
            "status": self.status,
            "version": self.version,
            "languages": list(self.languages),
            "needs_gpu": self.needs_gpu,
            "can_use_gpu": self.can_use_gpu,
            "models_on_disk": [m.to_dict() for m in self.models_on_disk],
            "notes": list(self.notes),
            "error": self.error,
            "location": self.location,
            "kind": self.kind,
            "install_hint": self.install_hint,
            "runnable": self.runnable,
        }


# --------------------------------------------------------------------------- lookups


class Finder:
    """Looks up modules and package metadata without importing anything.

    ``search_paths=None`` uses the running interpreter's full import system
    (so editable installs are seen); a list restricts the search to those
    folders, which is how the tests build a fake environment.
    """

    def __init__(self, search_paths: Optional[Sequence[str]] = None) -> None:
        self.search_paths = None if search_paths is None else [str(p) for p in search_paths]

    @staticmethod
    def refresh() -> None:
        """Forget cached folder listings, so an engine installed a moment ago is seen.

        The import system caches each folder's contents until the folder's
        modification time changes, and on some filesystems that clock is too
        coarse to notice a ``pip install`` in the same second.
        """
        importlib.invalidate_caches()

    def spec(self, module: str) -> Tuple[Optional[Any], Optional[str]]:
        """``(spec, None)``, ``(None, None)`` if absent, ``(None, error)`` if lookup broke."""
        try:
            if self.search_paths is None:
                return importlib.util.find_spec(module), None
            return importlib.machinery.PathFinder.find_spec(module, self.search_paths), None
        except Exception as exc:  # ValueError, ImportError from a broken finder
            return None, "{0}: {1}".format(type(exc).__name__, exc)

    def has(self, module: str) -> bool:
        spec, error = self.spec(module)
        return spec is not None and error is None

    def version(self, *dist_names: str) -> Optional[str]:
        """The installed version of the first distribution found, if any."""
        if _metadata is None:  # pragma: no cover
            return None
        for name in dist_names:
            for candidate in (name, name.replace("-", "_")):
                try:
                    if self.search_paths is None:
                        return _metadata.version(candidate)
                    found = list(_metadata.distributions(name=candidate, path=self.search_paths))
                    if found:
                        return found[0].version
                except Exception:  # PackageNotFoundError or corrupt metadata
                    continue
        return None


def _package_dir(spec: Any) -> Optional[str]:
    origin = getattr(spec, "origin", None)
    if origin and origin not in ("namespace", "built-in", "frozen") and os.path.exists(origin):
        return os.path.dirname(origin)
    return None


def _is_leftover(spec: Any) -> bool:
    """A folder with no ``__init__.py``: what an interrupted uninstall leaves."""
    origin = getattr(spec, "origin", None)
    return origin in (None, "namespace") and bool(getattr(spec, "submodule_search_locations", None))


def _probe_python_engine(
    name: str,
    module: str,
    dists: Sequence[str],
    deps: Sequence[str],
    finder: Finder,
) -> Tuple[Engine, Optional[Any]]:
    engine = Engine(name=name, install_hint=INSTALL_HINTS.get(name, ""))
    spec, lookup_error = finder.spec(module)
    engine.version = finder.version(*dists)
    if lookup_error:
        engine.installed = True
        engine.error = "looking up the {0} module failed ({1})".format(module, lookup_error)
        return engine, None
    if spec is None:
        if engine.version:
            engine.installed = True
            engine.error = (
                "package metadata for {0} {1} is present but the {2} module is missing "
                "(a partial install or uninstall); reinstall it with pip install "
                "--force-reinstall {0}".format(dists[0], engine.version, module)
            )
        return engine, None
    if _is_leftover(spec):
        if engine.version:
            engine.installed = True
            engine.error = "the {0} folder has no __init__.py; reinstall {1}".format(module, dists[0])
        else:
            engine.notes.append(
                "an empty {0} folder is on the import path (left by an uninstall?)".format(module)
            )
        return engine, None
    engine.installed = True
    engine.location = _package_dir(spec)
    missing = [dep for dep in deps if not finder.has(dep)]
    if missing:
        engine.error = (
            "its dependenc{0} {1} {2} missing, so importing it would fail; repair "
            "with pip install --force-reinstall {3}".format(
                "ies" if len(missing) > 1 else "y",
                ", ".join(missing),
                "are" if len(missing) > 1 else "is",
                dists[0],
            )
        )
    return engine, spec


def _cuda_platform() -> bool:
    return sys.platform == "win32" or sys.platform.startswith("linux")


def probe_faster_whisper(finder: Finder, scan: ModelScan) -> Engine:
    """faster-whisper: CTranslate2 underneath, int8 on the CPU, float16 on CUDA."""
    engine, _spec = _probe_python_engine(
        FASTER_WHISPER,
        "faster_whisper",
        ("faster-whisper",),
        ("ctranslate2", "av", "tokenizers", "huggingface_hub", "numpy"),
        finder,
    )
    engine.can_use_gpu = _cuda_platform()
    if engine.installed and _cuda_platform() and any(g.backend == "cuda" for g in detect_gpus()):
        engine.notes.append(
            "uses an NVIDIA GPU when the CUDA 12 cuBLAS and cuDNN 9 libraries are "
            "present; otherwise transcribe() falls back to the CPU"
        )
    engine.models_on_disk = scan.for_engine(FASTER_WHISPER)
    engine.notes.extend(scan.notes_for(FASTER_WHISPER))
    return engine.finish()


def _torch_has_cuda(finder: Finder) -> Optional[bool]:
    spec, _error = finder.spec("torch")
    root = _package_dir(spec) if spec is not None else None
    if root is None:
        return None
    lib = os.path.join(root, "lib")
    try:
        names = [n.lower() for n in os.listdir(lib)]
    except OSError:
        return None
    return any("cuda" in n or "cudart" in n for n in names)


def _openai_whisper_version_from_source(folder: str) -> Optional[str]:
    try:
        with open(os.path.join(folder, "version.py"), "r", encoding="utf-8") as handle:
            match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", handle.read())
        return match.group(1) if match else None
    except OSError:
        return None


def probe_openai_whisper(finder: Finder, scan: ModelScan) -> Engine:
    """openai-whisper (``import whisper``): PyTorch underneath."""
    engine, spec = _probe_python_engine(
        OPENAI_WHISPER,
        "whisper",
        ("openai-whisper",),
        ("torch", "numpy", "tiktoken", "numba"),
        finder,
    )
    if spec is not None and engine.version is None:
        folder = _package_dir(spec) or ""
        looks_right = all(os.path.exists(os.path.join(folder, f)) for f in ("transcribe.py", "audio.py"))
        if looks_right:
            engine.version = _openai_whisper_version_from_source(folder)
            engine.notes.append("found as source without package metadata")
        else:
            engine = Engine(name=OPENAI_WHISPER, install_hint=INSTALL_HINTS[OPENAI_WHISPER])
            engine.notes.append(
                "a different package named 'whisper' is installed (probably the "
                "Graphite time-series database); it is not openai-whisper"
            )
    if engine.installed and not engine.error:
        cuda = _torch_has_cuda(finder)
        engine.can_use_gpu = bool(cuda) and _cuda_platform()
        if cuda is False and _cuda_platform():
            engine.notes.append("PyTorch here is the CPU-only build, so openai-whisper cannot use a GPU")
        if shutil.which("ffmpeg") is None:
            engine.notes.append(
                "ffmpeg is not on PATH; openai-whisper needs it for MP3, M4A and similar "
                "files (WAV files are decoded by this package instead)"
            )
    elif not engine.installed:
        engine.can_use_gpu = _cuda_platform()
    engine.models_on_disk = scan.for_engine(OPENAI_WHISPER)
    engine.notes.extend(scan.notes_for(OPENAI_WHISPER))
    return engine.finish()


def find_whisper_cpp() -> Tuple[Optional[str], Optional[str]]:
    """``(binary, problem)`` - the whisper.cpp command line tool, never run."""
    explicit = os.environ.get(WHISPER_CPP_BIN_ENV, "").strip()
    if explicit:
        resolved = shutil.which(explicit) or (explicit if os.path.isfile(explicit) else None)
        if resolved is None:
            return None, "{0} points at {1}, which does not exist".format(WHISPER_CPP_BIN_ENV, explicit)
        return os.path.abspath(resolved), None
    for name in WHISPER_CPP_NAMES:
        found = shutil.which(name)
        if found:
            return os.path.abspath(found), None
    return None, None


def probe_whisper_cpp(finder: Finder, scan: ModelScan, binary: Optional[str], problem: Optional[str]) -> Engine:
    """whisper.cpp: a native binary. Found on PATH, never executed while probing."""
    engine = Engine(name=WHISPER_CPP, install_hint=INSTALL_HINTS[WHISPER_CPP])
    apple = is_apple_silicon()
    engine.can_use_gpu = apple
    if problem:
        engine.installed = True
        engine.error = problem
    elif binary:
        engine.installed = True
        engine.location = binary
        real = os.path.realpath(binary)
        match = re.search(r"whisper-cpp[\\/](\d+\.\d+(?:\.\d+)?)", real)
        engine.version = match.group(1) if match else None
        try:
            size = os.path.getsize(binary)
        except OSError:
            size = 0
        if size == 0:
            engine.error = "{0} is an empty file".format(binary)
        elif os.name != "nt" and not os.access(binary, os.X_OK):
            engine.error = "{0} is not executable (chmod +x it)".format(binary)
        if engine.version is None:
            engine.notes.append("version not read: probing never runs the binary")
        if not apple:
            engine.notes.append(
                "a CUDA or Vulkan build uses the GPU by itself; the router plans for the CPU"
            )
    engine.models_on_disk = scan.for_engine(WHISPER_CPP)
    engine.notes.extend(scan.notes_for(WHISPER_CPP))
    if engine.installed and not engine.models_on_disk:
        folders = whisper_cpp_dirs(binary)
        engine.notes.append("looked for ggml-*.bin in: " + os.pathsep.join(folders[:4]))
    return engine.finish()


_VOSK_LIBS = {"win32": "libvosk.dll", "darwin": "libvosk.dyld"}


def vosk_library_name(system: Optional[str] = None) -> str:
    """The native library the vosk package loads on this platform."""
    return _VOSK_LIBS.get(sys.platform if system is None else system, "libvosk.so")


def probe_vosk(finder: Finder, scan: ModelScan) -> Engine:
    """Vosk (Kaldi): small per-language models, CPU only, low memory."""
    engine, spec = _probe_python_engine(
        VOSK, "vosk", ("vosk",), ("_cffi_backend", "requests", "srt", "tqdm"), finder
    )
    if spec is not None and not engine.error:
        folder = _package_dir(spec)
        lib = vosk_library_name()
        if folder and not os.path.isfile(os.path.join(folder, lib)):
            engine.error = "its native library {0} is missing from {1}; reinstall vosk".format(lib, folder)
    engine.models_on_disk = scan.for_engine(VOSK)
    engine.notes.extend(scan.notes_for(VOSK))
    return engine.finish()


# --------------------------------------------------------------------------- custom


Probe = Callable[[], Any]

_CUSTOM_KEYS = {
    "installed", "version", "languages", "needs_gpu", "can_use_gpu", "uses_gpu",
    "models_on_disk", "models", "notes", "error", "quality", "realtime_factor",
    "ram_gb", "vram_gb", "needs_model", "location",
}


def _clean_languages(values: Any, notes: List[str]) -> List[str]:
    if values is None:
        return ["*"]
    if isinstance(values, str):
        values = [values]
    out: List[str] = []
    for value in values:
        if value == "*":
            code: Optional[str] = "*"
        else:
            try:
                code = normalize_language(value)
            except (ValueError, TypeError):
                notes.append("ignored language {0!r}: not recognised".format(value))
                continue
        if code is None:
            code = "*"
        if code not in out:
            out.append(code)
    return out or ["*"]


def _custom_models(name: str, raw: Any, languages: List[str], notes: List[str]) -> List[LocalModel]:
    models: List[LocalModel] = []
    for item in raw or []:
        if isinstance(item, LocalModel):
            models.append(replace(item, engine=name))
            continue
        if isinstance(item, (str, os.PathLike)):
            text = os.fspath(item)
            path = os.path.abspath(text) if os.path.exists(text) else ""
            label = os.path.basename(os.path.normpath(text)) if path else text
            models.append(LocalModel(name=label, engine=name, family="custom", path=path, languages=list(languages)))
            continue
        if isinstance(item, dict) and item.get("name"):
            model_langs = _clean_languages(item.get("languages"), notes) if "languages" in item else list(languages)
            models.append(
                LocalModel(
                    name=str(item["name"]),
                    engine=name,
                    family=str(item.get("family", "custom")),
                    path=str(item.get("path", "")),
                    size_gb=float(item.get("size_gb", 0.0) or 0.0),
                    languages=model_langs,
                    ram_gb_hint=_float_or_none(item.get("ram_gb")),
                    quality_hint=_unit_or_none(item.get("quality")),
                    realtime_factor_hint=_float_or_none(item.get("realtime_factor")),
                )
            )
            continue
        notes.append("ignored model entry {0!r}: expected a name, a path or a dict with 'name'".format(item))
    return models


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _unit_or_none(value: Any) -> Optional[float]:
    number = _float_or_none(value)
    if number is None:
        return None
    return max(0.0, min(1.0, number))


def run_custom_probe(name: str, probe: Probe, runnable: bool) -> Engine:
    """Call a registered probe and turn whatever it returns into an ``Engine``.

    A probe may return ``True``/``False``/``None``, a dict of Engine fields, or
    an ``Engine``. A probe that raises ``ModuleNotFoundError`` means "not
    installed"; any other exception means installed-but-failing. Nothing a
    probe does can crash ``available()``.
    """
    base = Engine(name=name, kind="custom", needs_model=False, runnable=runnable)
    try:
        result = probe()
    except ModuleNotFoundError as exc:
        base.notes.append("its probe could not import {0}".format(getattr(exc, "name", None) or exc))
        return base.finish()
    except Exception as exc:
        base.installed = True
        base.error = "its probe raised {0}: {1}".format(type(exc).__name__, exc)
        logger.warning("probe for engine %r raised %s", name, exc)
        return base.finish()
    if result is None or result is False:
        return base.finish()
    if result is True:
        base.installed = True
        base.languages = ["*"]
        return base.finish()
    if isinstance(result, Engine):
        engine = replace(result, name=name, kind="custom", runnable=runnable)
        engine.notes = list(engine.notes)
        if engine.needs_model and not engine.models_on_disk:
            engine.needs_model = False
        if not engine.languages:
            engine.languages = ["*"]
        languages = list(engine.languages)
        engine.finish()
        if not engine.needs_model:
            engine.languages = languages
        return engine
    if not isinstance(result, dict):
        base.installed = True
        base.error = "its probe returned {0}; expected True/False, a dict or an Engine".format(
            type(result).__name__
        )
        return base.finish()
    notes: List[str] = [str(n) for n in result.get("notes", []) or []]
    unknown = sorted(set(result) - _CUSTOM_KEYS)
    if unknown:
        notes.append("ignored unknown probe keys: " + ", ".join(unknown))
    languages = _clean_languages(result.get("languages"), notes)
    models = _custom_models(name, result.get("models_on_disk", result.get("models")), languages, notes)
    needs_model = bool(result.get("needs_model", bool(models)))
    engine = Engine(
        name=name,
        installed=bool(result.get("installed", True)),
        version=None if result.get("version") is None else str(result.get("version")),
        needs_gpu=bool(result.get("needs_gpu", False)),
        can_use_gpu=bool(result.get("can_use_gpu", result.get("uses_gpu", result.get("needs_gpu", False)))),
        models_on_disk=models,
        notes=notes,
        error=None if not result.get("error") else str(result.get("error")),
        kind="custom",
        needs_model=needs_model,
        runnable=runnable,
        location=None if result.get("location") is None else str(result.get("location")),
        quality=_unit_or_none(result.get("quality")),
        realtime_factor=_float_or_none(result.get("realtime_factor")),
        ram_gb=_float_or_none(result.get("ram_gb")),
        vram_gb=_float_or_none(result.get("vram_gb")),
    )
    engine.finish()
    if not engine.needs_model:
        engine.languages = languages
    return engine
