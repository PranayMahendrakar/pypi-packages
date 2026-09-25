"""Model files on disk: where each engine keeps them, and what each one is.

Nothing here downloads, imports an engine, or opens a network connection. A
model is identified from its own bytes where the format allows it:

* whisper.cpp ``ggml-*.bin`` files carry their dimensions in a 48-byte header,
  so the true size, the quantization and whether the model is English-only
  are read from the file rather than guessed from its name. A file shorter
  than its header says it should be is a partial download.
* openai-whisper ``.pt`` checkpoints are zip archives. A partial download has
  no zip directory at its end, and the model dimensions are read from the
  embedded pickle as opcodes, without unpickling anything.
* faster-whisper models are CTranslate2 folders, found in the Hugging Face
  cache or any folder you point at. A snapshot without ``model.bin`` is an
  unfinished download.
* Vosk models are folders with ``am/final.mdl`` and a ``conf`` folder.

The per-model RAM, speed and quality figures are rough, taken from the numbers
each project publishes. They rank models against each other; they are not
benchmarks of your machine.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickletools
import re
import struct
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._languages import (
    WHISPER_LANGUAGES,
    all_whisper_codes,
    vosk_language_from_name,
)

logger = logging.getLogger(__name__)

_GB = 1024.0 ** 3

FASTER_WHISPER = "faster-whisper"
OPENAI_WHISPER = "openai-whisper"
WHISPER_CPP = "whisper.cpp"
VOSK = "vosk"
WHISPER_ENGINES = (FASTER_WHISPER, OPENAI_WHISPER, WHISPER_CPP)

#: Folders to search for models of every kind, separated like PATH.
MODELS_ENV = "OFFLINE_STT_MODELS"
#: Extra folders for whisper.cpp ggml files, separated like PATH.
WHISPER_CPP_MODELS_ENV = "WHISPER_CPP_MODELS"


@dataclass(frozen=True)
class Family:
    """A Whisper model size and what it costs."""

    key: str
    params_m: float
    quality: float
    units: float
    size_class: str
    rank: int
    english_only: bool = False
    v3: bool = False


#: ``quality`` is a relative 0-1 rank loosely following published word error
#: rates on well-covered languages; ``units`` is compute relative to small.
FAMILIES: Dict[str, Family] = {
    "tiny": Family("tiny", 39, 0.35, 0.25, "tiny", 0),
    "base": Family("base", 74, 0.45, 0.45, "base", 1),
    "small": Family("small", 244, 0.62, 1.0, "small", 2),
    "medium": Family("medium", 769, 0.74, 2.8, "medium", 3),
    "large-v1": Family("large-v1", 1550, 0.78, 5.5, "large", 5),
    "large-v2": Family("large-v2", 1550, 0.84, 5.5, "large", 5),
    "large-v3": Family("large-v3", 1550, 0.88, 5.5, "large", 5, v3=True),
    "large-v3-turbo": Family("large-v3-turbo", 809, 0.86, 2.2, "turbo", 4, v3=True),
    "distil-large-v3.5": Family("distil-large-v3.5", 756, 0.85, 2.0, "distil-large", 4, True, True),
    "distil-large-v3": Family("distil-large-v3", 756, 0.84, 2.0, "distil-large", 4, True, True),
    "distil-large-v2": Family("distil-large-v2", 756, 0.82, 2.0, "distil-large", 4, True),
    "distil-medium": Family("distil-medium", 394, 0.72, 1.2, "distil-medium", 3, True),
    "distil-small": Family("distil-small", 166, 0.60, 0.5, "distil-small", 2, True),
}

#: RAM in GB on the CPU, by engine and size class (faster-whisper at int8).
RAM_CPU: Dict[str, Dict[str, float]] = {
    FASTER_WHISPER: {
        "tiny": 0.35, "base": 0.45, "small": 0.9, "medium": 2.0, "large": 3.6,
        "turbo": 1.9, "distil-large": 1.8, "distil-medium": 1.1, "distil-small": 0.6,
    },
    OPENAI_WHISPER: {
        "tiny": 1.0, "base": 1.1, "small": 2.1, "medium": 5.0, "large": 10.0,
        "turbo": 6.0, "distil-large": 6.0, "distil-medium": 3.0, "distil-small": 1.5,
    },
    WHISPER_CPP: {
        "tiny": 0.27, "base": 0.39, "small": 0.85, "medium": 2.1, "large": 3.9,
        "turbo": 1.7, "distil-large": 1.6, "distil-medium": 1.0, "distil-small": 0.5,
    },
}

#: GPU memory in GB (faster-whisper at float16; openai-whisper as its README says).
VRAM_GPU: Dict[str, Dict[str, float]] = {
    FASTER_WHISPER: {
        "tiny": 0.5, "base": 0.7, "small": 1.2, "medium": 2.6, "large": 4.5,
        "turbo": 2.5, "distil-large": 2.3, "distil-medium": 1.4, "distil-small": 0.8,
    },
    OPENAI_WHISPER: {
        "tiny": 1.0, "base": 1.0, "small": 2.0, "medium": 5.0, "large": 10.0,
        "turbo": 6.0, "distil-large": 6.0, "distil-medium": 3.0, "distil-small": 1.5,
    },
}

#: Host RAM an engine still needs while the model sits on the GPU.
HOST_RAM_WITH_GPU: Dict[str, float] = {FASTER_WHISPER: 1.2, OPENAI_WHISPER: 2.5}

#: Download sizes in GB, used only to describe a model you do not have yet.
DOWNLOAD_GB: Dict[str, float] = {
    "tiny": 0.08, "base": 0.15, "small": 0.48, "medium": 1.5, "large": 3.1,
    "turbo": 1.6, "distil-large": 1.5, "distil-medium": 0.8, "distil-small": 0.33,
}

#: Vosk: quality, seconds of compute per second of audio, RAM floor.
VOSK_FAMILIES: Dict[str, Tuple[float, float, float]] = {
    "vosk-small": (0.42, 0.08, 0.3),
    "vosk-lgraph": (0.52, 0.20, 1.0),
    "vosk-big": (0.60, 0.35, 2.0),
}

#: Bytes per weight for each ggml ``ftype`` (after ``% 1000``).
_GGML_FTYPES: Dict[int, Tuple[str, float]] = {
    0: ("f32", 4.0), 1: ("f16", 2.0), 2: ("q4_0", 0.5625), 3: ("q4_1", 0.625),
    4: ("q4_1", 0.625), 7: ("q8_0", 1.0625), 8: ("q5_0", 0.6875),
    9: ("q5_1", 0.75), 10: ("q2_k", 0.33), 11: ("q3_k", 0.43),
    12: ("q4_k", 0.5625), 13: ("q5_k", 0.6875), 14: ("q6_k", 0.82),
}
_GGML_MAGIC = 0x67676D6C
_WIDTH_TO_FAMILY = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}
_DIM_KEYS = (
    "n_mels", "n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head",
    "n_audio_layer", "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer",
)
_ENGLISH_VOCAB = 51864


@dataclass
class LocalModel:
    """One model an engine could load, found on disk (or described, if not)."""

    name: str
    engine: str
    family: str
    path: str = ""
    size_gb: float = 0.0
    languages: List[str] = field(default_factory=list)
    quantization: Optional[str] = None
    source: str = ""
    ram_gb_hint: Optional[float] = None
    on_disk: bool = True
    notes: List[str] = field(default_factory=list)
    quality_hint: Optional[float] = None
    realtime_factor_hint: Optional[float] = None

    def __str__(self) -> str:
        return self.name

    @property
    def multilingual(self) -> bool:
        """True when it covers more than one language."""
        return "*" in self.languages or len(self.languages) > 1

    def supports(self, language: Optional[str]) -> bool:
        """Can it transcribe this language (``None`` = auto-detect)?"""
        if language is None:
            return self.multilingual and self.engine != VOSK
        return "*" in self.languages or language in self.languages

    def describe_languages(self) -> str:
        """``"English only"``, ``"99 languages"`` and so on."""
        if "*" in self.languages:
            return "any language"
        if not self.languages:
            return "language unknown"
        if len(self.languages) == 1:
            code = self.languages[0]
            return "{0} only".format(WHISPER_LANGUAGES.get(code, code).title())
        return "{0} languages".format(len(self.languages))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "name": self.name,
            "engine": self.engine,
            "family": self.family,
            "path": self.path,
            "size_gb": round(self.size_gb, 3),
            "languages": list(self.languages),
            "quantization": self.quantization,
            "source": self.source,
            "on_disk": self.on_disk,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------- names


def family_from_name(text: str) -> Optional[str]:
    """``"faster-whisper-large-v3"`` -> ``"large-v3"``; ``"ggml-base.en"`` -> ``"base"``."""
    t = text.lower()
    if "distil" in t:
        if "large-v3.5" in t or "large-v3-5" in t:
            return "distil-large-v3.5"
        if "large-v3" in t:
            return "distil-large-v3"
        if "large" in t:
            return "distil-large-v2"
        if "medium" in t:
            return "distil-medium"
        if "small" in t:
            return "distil-small"
        return None
    if "turbo" in t:
        return "large-v3-turbo"
    match = re.search(r"large[-_]?v(\d)", t)
    if match:
        return "large-v{0}".format(min(3, max(1, int(match.group(1)))))
    if re.search(r"(^|[^a-z])large([^a-z]|$)", t):
        return "large-v3"
    match = re.search(r"(^|[^a-z])(tiny|base|small|medium)([^a-z]|$)", t)
    if match:
        return match.group(2)
    return None


def english_only_name(text: str) -> bool:
    """Names ending in ``.en`` (``base.en``, ``medium.en``) are English-only."""
    return bool(re.search(r"\.en($|[^a-z])", text.lower()))


def whisper_languages_for(family: Optional[str], english_only: bool) -> List[str]:
    """The language list a Whisper-family model covers."""
    fam = FAMILIES.get(family or "")
    if english_only or (fam is not None and fam.english_only):
        return ["en"]
    return list(all_whisper_codes(v3=bool(fam and fam.v3)))


def family_from_dims(dims: Dict[str, int], name: str) -> Optional[str]:
    """Identify a Whisper checkpoint from its tensor dimensions."""
    width = dims.get("n_audio_state")
    base = _WIDTH_TO_FAMILY.get(int(width)) if width else None
    if base is None:
        return None
    text_layers = int(dims.get("n_text_layer") or 0)
    v3 = int(dims.get("n_mels") or 80) == 128
    if base == "large":
        if text_layers == 4:
            return "large-v3-turbo"
        if text_layers == 2:
            return "distil-large-v3" if v3 else "distil-large-v2"
        if v3:
            return "large-v3"
        named = family_from_name(name)
        return named if named in ("large-v1", "large-v2") else "large-v2"
    if text_layers == 2 and base in ("medium", "small"):
        return "distil-" + base
    return base


def _nearest_family(params_m: float) -> str:
    best = min(
        (fam for fam in FAMILIES.values() if not fam.english_only),
        key=lambda fam: abs(fam.params_m - params_m),
    )
    return best.key


# --------------------------------------------------------------------------- where


def _home() -> str:
    return os.path.expanduser("~")


def _env_dirs(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    return [part for part in raw.split(os.pathsep) if part.strip()]


def hf_hub_dir() -> str:
    """The Hugging Face hub cache, resolved the way ``huggingface_hub`` does."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return os.environ[name]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        return os.path.join(os.environ["XDG_CACHE_HOME"], "huggingface", "hub")
    return os.path.join(_home(), ".cache", "huggingface", "hub")


def whisper_cache_dir() -> str:
    """Where openai-whisper saves checkpoints (``~/.cache/whisper``)."""
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(_home(), ".cache")
    return os.path.join(root, "whisper")


def vosk_dirs() -> List[str]:
    """The folders Vosk itself searches for models."""
    dirs = _env_dirs("VOSK_MODEL_PATH")
    dirs += [
        "/usr/share/vosk",
        os.path.join(_home(), "AppData", "Local", "vosk"),
        os.path.join(_home(), ".cache", "vosk"),
    ]
    return dirs


def whisper_cpp_dirs(binary: Optional[str] = None) -> List[str]:
    """Folders where whisper.cpp ggml models usually live."""
    dirs = _env_dirs(WHISPER_CPP_MODELS_ENV)
    if binary:
        here = os.path.dirname(os.path.abspath(binary))
        dirs += [
            os.path.join(here, "models"),
            os.path.join(here, "..", "models"),
            os.path.join(here, "..", "..", "models"),
            os.path.join(here, "..", "share", "whisper-cpp"),
            os.path.join(here, "..", "share", "whisper-cpp", "models"),
        ]
    dirs += [
        os.path.join(_home(), ".cache", "whisper.cpp"),
        os.path.join(_home(), "whisper.cpp", "models"),
        os.path.join(_home(), ".local", "share", "whisper.cpp"),
    ]
    return dirs


def extra_dirs(model_dirs: Sequence[str] = ()) -> List[str]:
    """``OFFLINE_STT_MODELS`` plus folders passed in code or on the CLI."""
    return _env_dirs(MODELS_ENV) + [str(d) for d in model_dirs]


def _listdir(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def _isdir(path: str) -> bool:
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += _size(os.path.join(root, name))
    return total


def _short(path: str) -> str:
    home = _home()
    try:
        if os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(home) + os.sep):
            return "~" + os.path.abspath(path)[len(home):]
    except ValueError:  # different drives on Windows
        pass
    return path


# --------------------------------------------------------------------------- ggml


def read_ggml_header(path: str) -> Optional[Dict[str, int]]:
    """The hyper-parameters in a whisper.cpp model's header, or ``None``."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(48)
    except OSError:
        return None
    if len(head) < 48 or struct.unpack("<I", head[:4])[0] != _GGML_MAGIC:
        return None
    values = struct.unpack("<11i", head[4:48])
    keys = (
        "n_vocab", "n_audio_ctx", "n_audio_state", "n_audio_head", "n_audio_layer",
        "n_text_ctx", "n_text_state", "n_text_head", "n_text_layer", "n_mels", "ftype",
    )
    return dict(zip(keys, values))


def _weight_count(dims: Dict[str, int]) -> float:
    da = float(dims.get("n_audio_state") or 0)
    dt = float(dims.get("n_text_state") or 0)
    la = float(dims.get("n_audio_layer") or 0)
    lt = float(dims.get("n_text_layer") or 0)
    vocab = float(dims.get("n_vocab") or 0)
    mels = float(dims.get("n_mels") or 80)
    return 12 * la * da * da + 16 * lt * dt * dt + vocab * dt + mels * da * 3 + da * da * 3


def inspect_ggml(path: str) -> Tuple[Optional[LocalModel], Optional[str]]:
    """A whisper.cpp model from its header, or a note saying why it is not one."""
    base = os.path.basename(path)
    size = _size(path)
    if size == 0:
        return None, "{0} is empty (a failed download?)".format(_short(path))
    dims = read_ggml_header(path)
    if dims is None:
        try:
            with open(path, "rb") as handle:
                magic = handle.read(4)
        except OSError:
            magic = b""
        if magic == b"GGUF":
            return None, "{0} is a GGUF file, not a whisper.cpp model".format(_short(path))
        return None, "{0} does not start with a whisper.cpp model header".format(_short(path))
    name = re.sub(r"^ggml-", "", re.sub(r"\.bin$", "", base))
    ftype = int(dims.get("ftype", 1)) % 1000
    quant, bytes_per_weight = _GGML_FTYPES.get(ftype, ("ftype{0}".format(ftype), 0.0))
    if bytes_per_weight:
        expected = _weight_count(dims) * bytes_per_weight
        if expected > 0 and size < 0.7 * expected:
            return None, (
                "{0} is cut short: {1:.0f} MB on disk, its header describes about "
                "{2:.0f} MB (an interrupted download)".format(
                    _short(path), size / 1e6, expected / 1e6
                )
            )
    family = family_from_dims(dims, name) or family_from_name(name)
    if family is None:
        family = _nearest_family(_weight_count(dims) / 1e6)
    english = int(dims.get("n_vocab") or 0) == _ENGLISH_VOCAB or english_only_name(name)
    if int(dims.get("n_vocab") or 0) >= _ENGLISH_VOCAB + 2:
        languages = list(all_whisper_codes(v3=True))
    else:
        languages = whisper_languages_for(family, english)
    size_gb = size / _GB
    model = LocalModel(
        name=name,
        engine=WHISPER_CPP,
        family=family,
        path=os.path.abspath(path),
        size_gb=size_gb,
        languages=languages,
        quantization=quant,
        source=_short(os.path.dirname(os.path.abspath(path))),
        ram_gb_hint=size_gb * 1.15 + 0.2,
    )
    return model, None


# --------------------------------------------------------------------------- .pt


def read_pt_dims(path: str) -> Tuple[Optional[Dict[str, int]], str]:
    """Whisper dimensions from a PyTorch checkpoint, without unpickling it.

    Returns ``(dims, status)`` where status is ``"ok"``, ``"no-dims"``,
    ``"legacy"`` (an old non-zip checkpoint that cannot be checked) or
    ``"truncated"`` (no zip directory: the download did not finish).
    """
    try:
        if not zipfile.is_zipfile(path):
            with open(path, "rb") as handle:
                start = handle.read(2)
            return None, ("legacy" if start[:1] == b"\x80" and _size(path) > 1_000_000 else "truncated")
        with zipfile.ZipFile(path) as archive:
            member = next((n for n in archive.namelist() if n.endswith("data.pkl")), None)
            if member is None:
                return None, "no-dims"
            info = archive.getinfo(member)
            if info.file_size > 64 * 1024 * 1024:
                return None, "no-dims"
            payload = archive.read(member)
    except (OSError, zipfile.BadZipFile, EOFError):
        return None, "truncated"
    dims: Dict[str, int] = {}
    pending: Optional[str] = None
    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(payload)):
            name = opcode.name
            if name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                pending = arg if arg in _DIM_KEYS else None
            elif name in ("BININT", "BININT1", "BININT2", "LONG1", "INT", "LONG"):
                if pending is not None and isinstance(arg, int) and pending not in dims:
                    dims[pending] = int(arg)
                pending = None
            elif name in ("BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT"):
                continue
            else:
                pending = None
    except Exception:  # a pickle this simple scanner cannot walk
        return (dims or None), ("ok" if dims else "no-dims")
    return (dims or None), ("ok" if "n_audio_state" in dims else "no-dims")


def inspect_pt(path: str) -> Tuple[Optional[LocalModel], Optional[str]]:
    """An openai-whisper checkpoint, or a note saying why it is not usable."""
    base = os.path.basename(path)
    name = re.sub(r"\.pt$", "", base)
    size = _size(path)
    if size == 0:
        return None, "{0} is empty (a failed download?)".format(_short(path))
    dims, status = read_pt_dims(path)
    if status == "truncated":
        return None, (
            "{0} is not a complete checkpoint (no zip directory at its end, which "
            "is what an interrupted download looks like)".format(_short(path))
        )
    notes: List[str] = []
    family = None
    english = english_only_name(name)
    if dims:
        family = family_from_dims(dims, name)
        english = int(dims.get("n_vocab") or 0) == _ENGLISH_VOCAB
    if family is None:
        family = family_from_name(name)
    if family is None:
        if status == "no-dims":
            return None, "{0} is a zip archive but not a Whisper checkpoint".format(_short(path))
        family = _nearest_family(size / _GB * 1000 / 2.0)
        notes.append("size family guessed from the file size")
    if status == "legacy":
        notes.append("old checkpoint format; could not be checked for completeness")
    languages = whisper_languages_for(family, english)
    if dims and int(dims.get("n_vocab") or 0) >= _ENGLISH_VOCAB + 2 and not english:
        languages = list(all_whisper_codes(v3=True))
    return (
        LocalModel(
            name=name,
            engine=OPENAI_WHISPER,
            family=family,
            path=os.path.abspath(path),
            size_gb=size / _GB,
            languages=languages,
            quantization="f16",
            source=_short(os.path.dirname(os.path.abspath(path))),
            notes=notes,
        ),
        None,
    )


# --------------------------------------------------------------------------- CTranslate2


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def inspect_ct2(path: str, label: Optional[str] = None, source: str = "") -> Tuple[Optional[LocalModel], Optional[str]]:
    """A faster-whisper (CTranslate2) model folder, or a note."""
    label = label or os.path.basename(os.path.normpath(path))
    model_bin = os.path.join(path, "model.bin")
    config_path = os.path.join(path, "config.json")
    config = _read_json(config_path) if os.path.exists(config_path) else None
    if config is not None and ("model_type" in config or "architectures" in config):
        return None, (
            "{0} is a Hugging Face Transformers checkpoint, not a CTranslate2 model; "
            "faster-whisper can use it only after conversion with "
            "ct2-transformers-converter".format(label)
        )
    if os.path.islink(model_bin) and not os.path.exists(model_bin):
        return None, "{0}: model.bin points at a missing file (an interrupted download)".format(label)
    if not os.path.isfile(model_bin):
        if config is not None or os.path.exists(os.path.join(path, "tokenizer.json")):
            return None, "{0}: download not finished (no model.bin yet)".format(label)
        return None, None
    size = _size(model_bin)
    if size == 0:
        return None, "{0}: model.bin is empty (an interrupted download)".format(label)
    family = family_from_name(label)
    notes: List[str] = []
    if family is None:
        family = _nearest_family(size / _GB * 1000 / 2.0)
        notes.append("size family guessed from the file size")
    english = english_only_name(label)
    languages = whisper_languages_for(family, english)
    lang_ids = (config or {}).get("lang_ids")
    if isinstance(lang_ids, list) and not english and not FAMILIES[family].english_only:
        if len(lang_ids) >= 100:
            languages = list(all_whisper_codes(v3=True))
        elif len(lang_ids) >= 99:
            languages = list(all_whisper_codes(v3=False))
    return (
        LocalModel(
            name=label,
            engine=FASTER_WHISPER,
            family=family,
            path=os.path.abspath(path),
            size_gb=size / _GB,
            languages=languages,
            quantization=None,
            source=source or _short(os.path.dirname(os.path.abspath(path))),
            notes=notes,
        ),
        None,
    )


_KNOWN_REPOS = {
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "large-v3-turbo",
    "deepdml/faster-whisper-large-v3-turbo-ct2": "large-v3-turbo",
    "distil-whisper/distil-large-v3.5-ct2": "distil-large-v3.5",
}


def _repo_label(org: str, name: str) -> str:
    repo = "{0}/{1}".format(org, name)
    if repo in _KNOWN_REPOS:
        return _KNOWN_REPOS[repo]
    if org == "Systran":
        if name.startswith("faster-distil-whisper-"):
            return "distil-" + name[len("faster-distil-whisper-"):]
        if name.startswith("faster-whisper-"):
            return name[len("faster-whisper-"):]
    return repo


def _snapshot(repo_dir: str) -> Optional[str]:
    snapshots = os.path.join(repo_dir, "snapshots")
    ref = os.path.join(repo_dir, "refs", "main")
    try:
        with open(ref, "r", encoding="utf-8") as handle:
            revision = handle.read().strip()
        candidate = os.path.join(snapshots, revision)
        if revision and _isdir(candidate):
            return candidate
    except OSError:
        pass
    revisions = [os.path.join(snapshots, r) for r in _listdir(snapshots)]
    revisions = [r for r in revisions if _isdir(r)]
    if not revisions:
        return None
    return max(revisions, key=lambda r: os.path.getmtime(r))


# --------------------------------------------------------------------------- Vosk


def _is_vosk_dir(path: str) -> bool:
    am = os.path.isfile(os.path.join(path, "am", "final.mdl")) or os.path.isfile(
        os.path.join(path, "final.mdl")
    )
    conf = any(
        os.path.isfile(os.path.join(path, *parts))
        for parts in (("conf", "mfcc.conf"), ("conf", "model.conf"), ("mfcc.conf",))
    )
    return am and conf


def inspect_vosk(path: str) -> Tuple[Optional[LocalModel], Optional[str]]:
    """A Vosk model folder, or a note saying what is wrong with it."""
    name = os.path.basename(os.path.normpath(path))
    if not _is_vosk_dir(path):
        return None, None
    has_graph = _isdir(os.path.join(path, "graph")) or any(
        entry.endswith(".fst") for entry in _listdir(path)
    )
    if not has_graph:
        return None, "{0}: no decoding graph (graph/ or *.fst) - the unzip looks incomplete".format(
            _short(path)
        )
    lowered = name.lower()
    if "small" in lowered:
        family = "vosk-small"
    elif "lgraph" in lowered:
        family = "vosk-lgraph"
    else:
        family = "vosk-big"
    size_gb = _dir_size(path) / _GB
    language = vosk_language_from_name(name)
    notes: List[str] = []
    if language is None:
        notes.append(
            "the folder name does not say which language this is, so it is only "
            "used when named explicitly (rename it like vosk-model-small-en-us-0.15)"
        )
    floor = VOSK_FAMILIES[family][2]
    ram = max(floor, size_gb * (1.2 if family == "vosk-small" else 2.5))
    return (
        LocalModel(
            name=name,
            engine=VOSK,
            family=family,
            path=os.path.abspath(path),
            size_gb=size_gb,
            languages=[language] if language else [],
            source=_short(os.path.dirname(os.path.abspath(path))),
            ram_gb_hint=ram,
            notes=notes,
        ),
        None,
    )


# --------------------------------------------------------------------------- scan


@dataclass
class ModelScan:
    """Every model found, grouped by engine, plus notes on what was skipped."""

    models: Dict[str, List[LocalModel]] = field(default_factory=dict)
    notes: Dict[str, List[str]] = field(default_factory=dict)
    _seen: set = field(default_factory=set)

    def add(self, model: Optional[LocalModel], note: Optional[str], engine: str) -> None:
        if note:
            bucket = self.notes.setdefault(engine, [])
            if note not in bucket:
                bucket.append(note)
        if model is None:
            return
        key = (model.engine, os.path.normcase(os.path.realpath(model.path)))
        if key in self._seen:
            return
        self._seen.add(key)
        self.models.setdefault(model.engine, []).append(model)

    def for_engine(self, engine: str) -> List[LocalModel]:
        return list(self.models.get(engine, []))

    def notes_for(self, engine: str) -> List[str]:
        return list(self.notes.get(engine, []))


def _scan_hf_hub(scan: ModelScan) -> None:
    hub = hf_hub_dir()
    for entry in _listdir(hub):
        if not entry.startswith("models--"):
            continue
        parts = entry[len("models--"):].split("--", 1)
        if len(parts) != 2:
            continue
        org, name = parts
        repo_dir = os.path.join(hub, entry)
        if org == "ggerganov" and name == "whisper.cpp":
            snapshot = _snapshot(repo_dir)
            if snapshot:
                _scan_ggml_dir(scan, snapshot)
            continue
        if "whisper" not in name.lower():
            continue
        snapshot = _snapshot(repo_dir)
        label = _repo_label(org, name)
        if snapshot is None:
            scan.add(None, "{0}: download not finished (no snapshot yet)".format(label), FASTER_WHISPER)
            continue
        if label in {m.name for m in scan.for_engine(FASTER_WHISPER)}:
            label = "{0}/{1}".format(org, name)
        model, note = inspect_ct2(snapshot, label=label, source="Hugging Face cache")
        if model is None and note is None:
            files = _listdir(snapshot)
            if any(f in files for f in ("pytorch_model.bin", "model.safetensors")):
                note = (
                    "{0}/{1} is a Hugging Face Transformers checkpoint, not a CTranslate2 "
                    "model; faster-whisper can use it only after conversion".format(org, name)
                )
        scan.add(model, note, FASTER_WHISPER)


def _scan_ggml_dir(scan: ModelScan, folder: str) -> None:
    for entry in _listdir(folder):
        lowered = entry.lower()
        if not (lowered.startswith("ggml-") and lowered.endswith(".bin")):
            continue
        path = os.path.join(folder, entry)
        if not os.path.isfile(path):
            continue
        model, note = inspect_ggml(path)
        scan.add(model, note, WHISPER_CPP)


def _scan_pt_dir(scan: ModelScan, folder: str) -> None:
    for entry in _listdir(folder):
        if not entry.lower().endswith(".pt"):
            continue
        path = os.path.join(folder, entry)
        if os.path.isfile(path):
            model, note = inspect_pt(path)
            scan.add(model, note, OPENAI_WHISPER)


def _scan_vosk_root(scan: ModelScan, root: str) -> None:
    if not _isdir(root):
        return
    candidates = [root] + [os.path.join(root, e) for e in _listdir(root)]
    for candidate in candidates:
        if _isdir(candidate):
            model, note = inspect_vosk(candidate)
            scan.add(model, note, VOSK)
        elif candidate.lower().endswith(".zip") and "vosk-model" in os.path.basename(candidate).lower():
            unzipped = candidate[:-4]
            if not _isdir(unzipped):
                scan.add(
                    None,
                    "{0} is still zipped; unzip it next to itself to use it".format(_short(candidate)),
                    VOSK,
                )


def _scan_mixed_dir(scan: ModelScan, folder: str) -> None:
    """A folder named by the user: look for every kind of model in it."""
    if not _isdir(folder):
        scan.add(None, "model folder {0} does not exist".format(_short(folder)), "any")
        return
    _scan_ggml_dir(scan, folder)
    _scan_pt_dir(scan, folder)
    for candidate in [folder] + [os.path.join(folder, e) for e in _listdir(folder)]:
        if not _isdir(candidate):
            continue
        if os.path.exists(os.path.join(candidate, "model.bin")) or os.path.exists(
            os.path.join(candidate, "config.json")
        ):
            model, note = inspect_ct2(candidate)
            scan.add(model, note, FASTER_WHISPER)
    _scan_vosk_root(scan, folder)


def scan_models(model_dirs: Sequence[str] = (), whisper_cpp_binary: Optional[str] = None) -> ModelScan:
    """Look in every standard place, plus ``model_dirs`` and ``OFFLINE_STT_MODELS``."""
    scan = ModelScan()
    _scan_hf_hub(scan)
    _scan_pt_dir(scan, whisper_cache_dir())
    for folder in whisper_cpp_dirs(whisper_cpp_binary):
        _scan_ggml_dir(scan, folder)
    for folder in vosk_dirs():
        _scan_vosk_root(scan, folder)
    for folder in extra_dirs(model_dirs):
        _scan_mixed_dir(scan, folder)
    return scan


def inspect_path(path: str) -> Tuple[Optional[LocalModel], Optional[str]]:
    """Whatever model lives at ``path``: a ggml file, a .pt file or a folder."""
    if os.path.isfile(path):
        lowered = path.lower()
        if lowered.endswith(".pt"):
            return inspect_pt(path)
        return inspect_ggml(path)
    if _isdir(path):
        if _is_vosk_dir(path):
            return inspect_vosk(path)
        return inspect_ct2(path)
    return None, "{0} does not exist".format(path)


# --------------------------------------------------------------------------- catalog

#: Models each engine can fetch by name, used only to suggest what to get.
CATALOG: Dict[str, Tuple[str, ...]] = {
    FASTER_WHISPER: (
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
        "medium.en", "large-v3", "large-v3-turbo", "distil-large-v3",
    ),
    OPENAI_WHISPER: (
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
        "medium.en", "large-v3", "turbo",
    ),
    WHISPER_CPP: (
        "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
        "medium.en", "large-v3", "large-v3-turbo",
    ),
}


def catalog_models(engine: str) -> List[LocalModel]:
    """Models ``engine`` could use once fetched (``on_disk=False``)."""
    out: List[LocalModel] = []
    for name in CATALOG.get(engine, ()):
        family = family_from_name(name) or "small"
        english = english_only_name(name) or FAMILIES[family].english_only
        out.append(
            LocalModel(
                name=name,
                engine=engine,
                family=family,
                size_gb=DOWNLOAD_GB.get(FAMILIES[family].size_class, 0.5),
                languages=whisper_languages_for(family, english),
                on_disk=False,
            )
        )
    return out


def vosk_catalog_model(language: Optional[str]) -> Optional[LocalModel]:
    """A stand-in for "a small Vosk model for this language", if one exists."""
    from ._languages import VOSK_LANGUAGES, language_name

    if language is None or language not in VOSK_LANGUAGES:
        return None
    return LocalModel(
        name="a small {0} model".format(language_name(language)),
        engine=VOSK,
        family="vosk-small",
        size_gb=0.05,
        languages=[language],
        ram_gb_hint=0.3,
        on_disk=False,
    )


def iter_models(scan: ModelScan, engines: Iterable[str]) -> List[LocalModel]:
    """Flatten a scan for the given engines."""
    out: List[LocalModel] = []
    for engine in engines:
        out.extend(scan.for_engine(engine))
    return out
