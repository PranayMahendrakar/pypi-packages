"""Weighing what is installed against the machine and the language.

The rule is deliberately simple enough to state in one sentence per mode:

* ``fast`` - the quickest option that is good enough for the language.
* ``balanced`` - the most accurate option expected to run at least 2x faster
  than real time on this hardware; if nothing does, the fastest.
* ``accurate`` - the most accurate option that fits in memory, whatever the
  speed.

"Good enough", "accurate" and "fast" come from rough per-model figures (see
``_models``), scaled by the CPU core count or the GPU. They are estimates for
ranking, not measurements of this machine.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._engines import INSTALL_HINTS, READY, Engine
from ._languages import language_name, normalize_language, whisper_tier
from ._machine import Machine
from ._models import (
    FAMILIES,
    FASTER_WHISPER,
    HOST_RAM_WITH_GPU,
    OPENAI_WHISPER,
    RAM_CPU,
    VOSK,
    VOSK_FAMILIES,
    VRAM_GPU,
    WHISPER_CPP,
    WHISPER_ENGINES,
    LocalModel,
    _short,
    catalog_models,
    vosk_catalog_model,
    whisper_cpp_dirs,
)

PREFERENCES = ("fast", "balanced", "accurate")

#: "balanced" wants at least this: 0.5 s of compute per second of audio (2x real time).
BALANCED_MAX_RTF = 0.5
_REFERENCE_THREADS = 8

#: Seconds of compute per second of audio for the *small* model, on an 8-core
#: CPU or a mid-range GPU, from each project's published benchmarks.
BASE_RTF: Dict[Tuple[str, str], float] = {
    (FASTER_WHISPER, "cpu"): 0.13,
    (FASTER_WHISPER, "cuda"): 0.013,
    (OPENAI_WHISPER, "cpu"): 0.55,
    (OPENAI_WHISPER, "cuda"): 0.035,
    (WHISPER_CPP, "cpu"): 0.18,
    (WHISPER_CPP, "metal"): 0.035,
}

#: Accuracy lost on languages Whisper saw less of in training.
_TIER_PENALTY: Dict[str, Dict[str, float]] = {
    "moderate": {"tiny": 0.12, "base": 0.10, "small": 0.07, "medium": 0.04, "large": 0.02, "turbo": 0.04},
    "limited": {"tiny": 0.25, "base": 0.22, "small": 0.15, "medium": 0.08, "large": 0.04, "turbo": 0.07},
}
#: The smallest model counted as good enough, by tier (rank: tiny 0, base 1, small 2).
_FLOOR_RANK = {"strong": 0, "moderate": 1, "limited": 2, "unsupported": 99}
_RANK_NAME = {0: "tiny", 1: "base", 2: "small"}
_QUANT = {"q8": (0.005, 0.9), "q6": (0.008, 0.88), "q5": (0.01, 0.85), "q4": (0.02, 0.8), "q3": (0.04, 0.8), "q2": (0.06, 0.8)}


@dataclass
class Option:
    """One way to run one model: an engine, a model, a device, and its estimates."""

    engine: str
    model: Optional[str]
    device: str
    compute_type: Optional[str] = None
    model_path: Optional[str] = None
    family: Optional[str] = None
    quality: float = 0.5
    realtime_factor: float = 0.3
    ram_gb: float = 1.0
    vram_gb: Optional[float] = None
    meets_floor: bool = True
    on_disk: bool = True
    reason: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def speed(self) -> str:
        """``"about 8x faster than real time"`` and similar."""
        return describe_speed(self.realtime_factor)

    def label(self) -> str:
        """``"faster-whisper small (CPU, int8)"``."""
        name = self.engine if not self.model else "{0} {1}".format(self.engine, self.model)
        device = self.device.upper() if self.device in ("cpu", "gpu") else _DEVICE_NAMES.get(self.device, self.device)
        extra = ", " + self.compute_type if self.compute_type else ""
        return "{0} ({1}{2})".format(name, device, extra)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "engine": self.engine,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_path": self.model_path,
            "family": self.family,
            "quality": round(self.quality, 3),
            "realtime_factor": round(self.realtime_factor, 4),
            "ram_gb": round(self.ram_gb, 2),
            "vram_gb": None if self.vram_gb is None else round(self.vram_gb, 2),
            "meets_floor": self.meets_floor,
            "on_disk": self.on_disk,
            "reason": self.reason,
            "notes": list(self.notes),
        }


_DEVICE_NAMES = {"cuda": "NVIDIA GPU", "metal": "Apple GPU"}


def describe_speed(rtf: float) -> str:
    """Plain words for a real-time factor."""
    if rtf <= 0:
        return "speed unknown"
    if rtf < 1.0:
        times = 1.0 / rtf
        return "about {0}x faster than real time".format(
            "{0:.0f}".format(times) if times >= 10 else "{0:.1f}".format(times)
        )
    return "about {0:.1f}x slower than real time".format(rtf)


@dataclass
class Choice:
    """Which engine and model to use, and why, in plain language.

    ``engine`` is ``None`` when nothing usable is installed; ``reason`` then
    says why and ``missing`` says what to install. Never raised, always returned.
    """

    engine: Optional[str]
    model: Optional[str]
    device: Optional[str]
    reason: str
    alternatives: List[Option] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    language: Optional[str] = None
    prefer: str = "balanced"
    compute_type: Optional[str] = None
    model_path: Optional[str] = None
    estimated_ram_gb: Optional[float] = None
    estimated_vram_gb: Optional[float] = None
    realtime_factor: Optional[float] = None
    rejected: List[Option] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    ram_budget_gb: Optional[float] = None
    machine: Optional[Machine] = None
    engines: List[Engine] = field(default_factory=list)
    option: Optional[Option] = None

    @property
    def ok(self) -> bool:
        """True when there is something to transcribe with."""
        return self.engine is not None

    def __bool__(self) -> bool:
        return self.ok

    @property
    def language_name(self) -> str:
        return language_name(self.language)

    @property
    def speed(self) -> Optional[str]:
        return None if self.realtime_factor is None else describe_speed(self.realtime_factor)

    def headline(self) -> str:
        """The first line of the summary."""
        if not self.ok or self.option is None:
            return "No speech-to-text engine is ready on this machine for {0}.".format(self.language_name)
        return "Use {0} for {1}.".format(_spoken(self.option), self.language_name)

    def summary(self) -> str:
        """Human-readable text, plain ASCII punctuation."""
        lines = [self.headline(), "Why: " + self.reason]
        if self.ok and self.option is not None:
            ram = "about {0:.1f} GB of RAM".format(self.option.ram_gb)
            if self.option.vram_gb is not None:
                ram += " and {0:.1f} GB of GPU memory".format(self.option.vram_gb)
            lines.append("Estimated: {0}, {1}.".format(self.option.speed, ram))
            if self.model_path:
                lines.append("Model: " + _short(self.model_path))
        for note in self.notes:
            lines.append("Note: " + note)
        if self.alternatives:
            lines.append("Alternatives:")
            for index, alt in enumerate(self.alternatives, 1):
                lines.append("  {0}. {1} - {2}".format(index, alt.label(), alt.reason))
        if self.rejected:
            lines.append("Set aside:")
            for rej in self.rejected[:8]:
                lines.append("  - {0}: {1}".format(rej.label() if rej.model else rej.engine, rej.reason))
            if len(self.rejected) > 8:
                lines.append("  - and {0} more".format(len(self.rejected) - 8))
        if self.missing:
            lines.append(
                "For a better result:" if self.ok else
                "To get one (this package never downloads anything itself):"
            )
            for item in self.missing:
                lines.append("  - " + item)
        if self.engines:
            lines.append("Checked: " + "; ".join(_engine_state(e) for e in self.engines) + ".")
        if self.machine is not None:
            lines.append("Machine: " + self.machine.describe())
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form."""
        return {
            "ok": self.ok,
            "engine": self.engine,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_path": self.model_path,
            "language": self.language,
            "language_name": self.language_name,
            "prefer": self.prefer,
            "reason": self.reason,
            "estimated_ram_gb": None if self.estimated_ram_gb is None else round(self.estimated_ram_gb, 2),
            "estimated_vram_gb": None if self.estimated_vram_gb is None else round(self.estimated_vram_gb, 2),
            "realtime_factor": None if self.realtime_factor is None else round(self.realtime_factor, 4),
            "speed": self.speed,
            "alternatives": [a.to_dict() for a in self.alternatives],
            "rejected": [r.to_dict() for r in self.rejected],
            "missing": list(self.missing),
            "notes": list(self.notes),
            "ram_budget_gb": None if self.ram_budget_gb is None else round(self.ram_budget_gb, 2),
            "machine": None if self.machine is None else self.machine.to_dict(),
            "engines": [
                {
                    "name": e.name,
                    "status": e.status,
                    "installed": e.installed,
                    "version": e.version,
                    "models_on_disk": [m.name for m in e.models_on_disk],
                    "error": e.error,
                }
                for e in self.engines
            ],
        }


def _spoken(option: Option) -> str:
    device = {"cpu": "the CPU", "cuda": "the NVIDIA GPU", "metal": "the Apple GPU", "gpu": "the GPU"}.get(
        option.device, option.device
    )
    compute = " ({0})".format(option.compute_type) if option.compute_type else ""
    if option.model:
        return "{0} with the {1} model on {2}{3}".format(option.engine, option.model, device, compute)
    return "{0} on {1}{2}".format(option.engine, device, compute)


def _engine_state(engine: Engine) -> str:
    state = {
        "ready": "ready",
        "no-model": "installed, no model",
        "failing": "installed but failing",
        "not-installed": "not installed",
    }.get(engine.status, engine.status)
    return "{0} ({1})".format(engine.name, state)


# --------------------------------------------------------------------------- estimates


def cpu_factor(machine: Machine) -> float:
    """How much slower (>1) or faster (<1) this CPU is than the 8-thread reference."""
    return (float(_REFERENCE_THREADS) / machine.threads) ** 0.7


def _quant_key(quantization: Optional[str]) -> Optional[str]:
    if not quantization:
        return None
    head = quantization.lower()[:2]
    return head if head in _QUANT else None


def model_quality(model: LocalModel, language: Optional[str]) -> float:
    """Relative accuracy (0-1) of a model for a language. A heuristic rank."""
    if model.quality_hint is not None:
        return model.quality_hint
    if model.engine == VOSK or model.family in VOSK_FAMILIES:
        return VOSK_FAMILIES.get(model.family, VOSK_FAMILIES["vosk-small"])[0]
    family = FAMILIES.get(model.family)
    if family is None:
        return 0.5
    quality = family.quality
    if language == "en" and not model.multilingual and family.size_class in ("tiny", "base", "small", "medium"):
        quality += 0.03
    tier = whisper_tier(language)
    if tier in _TIER_PENALTY:
        quality -= _TIER_PENALTY[tier].get(family.size_class, 0.04)
    key = _quant_key(model.quantization)
    if key:
        quality -= _QUANT[key][0]
    return max(0.0, min(1.0, quality))


def _meets_floor(model: LocalModel, language: Optional[str]) -> bool:
    family = FAMILIES.get(model.family)
    if family is None:
        return True
    return family.rank >= _FLOOR_RANK.get(whisper_tier(language), 0)


def floor_name(language: Optional[str]) -> Optional[str]:
    """The smallest Whisper size counted as good enough for a language."""
    return _RANK_NAME.get(_FLOOR_RANK.get(whisper_tier(language), 0))


def _variants(engine: Engine, model: Optional[LocalModel], machine: Machine, language: Optional[str]) -> List[Option]:
    """Every device this engine could run this model on, with estimates."""
    path = model.path if model is not None and model.path else None
    name = model.name if model is not None else None
    on_disk = True if model is None else model.on_disk
    family_key = model.family if model is not None else None

    if engine.name == VOSK and model is not None:
        quality, rtf, floor = VOSK_FAMILIES.get(model.family, VOSK_FAMILIES["vosk-small"])
        ram = model.ram_gb_hint if model.ram_gb_hint is not None else floor
        return [Option(VOSK, name, "cpu", None, path, family_key, quality, rtf, ram, None, True, on_disk)]

    family = FAMILIES.get(family_key or "")
    if engine.name not in WHISPER_ENGINES or model is None or family is None:
        quality = model_quality(model, language) if model is not None and model.quality_hint is not None else (
            engine.quality if engine.quality is not None else 0.5
        )
        rtf = (model.realtime_factor_hint if model is not None and model.realtime_factor_hint else None) or (
            engine.realtime_factor if engine.realtime_factor else 0.3
        )
        ram = (model.ram_gb_hint if model is not None and model.ram_gb_hint is not None else None)
        if ram is None:
            ram = engine.ram_gb if engine.ram_gb is not None else 1.0
        gpu = machine.cuda_gpu or machine.metal_gpu
        device = "gpu" if (engine.needs_gpu or (engine.can_use_gpu and gpu is not None)) else "cpu"
        return [Option(engine.name, name, device, None, path, family_key, quality, rtf, ram,
                       engine.vram_gb if device == "gpu" else None, True, on_disk)]

    quality = model_quality(model, language)
    floor_ok = _meets_floor(model, language)
    size = family.size_class
    units = family.units
    quant = _quant_key(model.quantization)
    speedup = _QUANT[quant][1] if quant else 1.0
    options: List[Option] = []

    cpu_ram = RAM_CPU[engine.name].get(size, 2.0)
    if engine.name == WHISPER_CPP and model.ram_gb_hint is not None:
        cpu_ram = model.ram_gb_hint
    compute = {FASTER_WHISPER: "int8", OPENAI_WHISPER: "float32", WHISPER_CPP: model.quantization or "f16"}[engine.name]
    options.append(Option(
        engine.name, name, "cpu", compute, path, family.key, quality,
        BASE_RTF[(engine.name, "cpu")] * units * speedup * cpu_factor(machine),
        cpu_ram, None, floor_ok, on_disk,
    ))
    if engine.can_use_gpu and machine.cuda_gpu is not None and engine.name in VRAM_GPU:
        vram = VRAM_GPU[engine.name].get(size, 3.0)
        host = min(cpu_ram, HOST_RAM_WITH_GPU[engine.name])
        rtf = BASE_RTF[(engine.name, "cuda")] * units
        if engine.name == FASTER_WHISPER:
            options.append(Option(engine.name, name, "cuda", "float16", path, family.key, quality, rtf,
                                  host, vram, floor_ok, on_disk))
            options.append(Option(engine.name, name, "cuda", "int8_float16", path, family.key,
                                  max(0.0, quality - 0.005), rtf * 1.1, host, vram * 0.65, floor_ok, on_disk))
        else:
            options.append(Option(engine.name, name, "cuda", "float16", path, family.key, quality, rtf,
                                  host, vram, floor_ok, on_disk))
    if engine.name == WHISPER_CPP and engine.can_use_gpu and machine.metal_gpu is not None:
        options.append(Option(
            engine.name, name, "metal", model.quantization or "f16", path, family.key, quality,
            BASE_RTF[(WHISPER_CPP, "metal")] * units * speedup, cpu_ram, None, floor_ok, on_disk,
        ))
    return options


# --------------------------------------------------------------------------- feasibility


def ram_budget(machine: Machine, max_ram_gb: Optional[float]) -> Tuple[Optional[float], str]:
    """The RAM a model may use, and how to say where the limit came from."""
    if max_ram_gb is not None:
        if isinstance(max_ram_gb, bool) or not isinstance(max_ram_gb, (int, float)):
            raise TypeError("max_ram_gb must be a number of gigabytes or None")
        if max_ram_gb <= 0:
            raise ValueError("max_ram_gb must be positive, got {0}".format(max_ram_gb))
    available = machine.ram_available_gb
    if max_ram_gb is not None and (available is None or max_ram_gb <= available):
        return float(max_ram_gb), "you allowed {0:.1f} GB (max_ram_gb)".format(float(max_ram_gb))
    if available is not None:
        text = "only {0:.1f} GB is available right now".format(available)
        if machine.ram_total_gb is not None:
            text += " (of {0:.1f} GB)".format(machine.ram_total_gb)
        return float(available), text
    return None, "RAM is unknown"


def _language_problem(model: Optional[LocalModel], engine: Engine, language: Optional[str]) -> Optional[str]:
    if model is None:
        if "*" in engine.languages or not engine.languages:
            return None
        if language is None:
            return None
        if language in engine.languages:
            return None
        return "does not list {0} among its languages".format(language_name(language))
    if model.supports(language):
        return None
    if language is None:
        if model.engine == VOSK:
            return "Vosk models cannot detect the language; pass language="
        return "{0}, so it cannot detect the language".format(model.describe_languages())
    if not model.languages:
        return "its language is unknown (name the model explicitly to use it)"
    return "{0}, not {1}".format(model.describe_languages(), language_name(language))


def _evaluate(
    engine: Engine,
    model: Optional[LocalModel],
    machine: Machine,
    language: Optional[str],
    budget: Optional[float],
    budget_text: str,
    ignore_ram: bool = False,
) -> Tuple[Optional[Option], Optional[Option]]:
    """``(best feasible variant, None)`` or ``(None, rejected variant with reason)``."""
    variants = _variants(engine, model, machine, language)
    problem = _language_problem(model, engine, language)
    if problem:
        rejected = variants[0]
        rejected.reason = problem
        return None, rejected
    feasible: List[Option] = []
    gpu_notes: List[str] = []
    rejected_ram: Optional[Option] = None
    for option in variants:
        if option.device in ("cuda", "gpu") and option.vram_gb is not None:
            gpu = machine.cuda_gpu if option.device == "cuda" else (machine.cuda_gpu or machine.metal_gpu)
            usable = gpu.usable_vram_gb if gpu is not None else None
            if gpu is None:
                continue
            if usable is not None and option.vram_gb > usable:
                gpu_notes.append(
                    "{0} does not fit in the GPU's {1:.1f} GB of free memory with {2} "
                    "(needs about {3:.1f} GB)".format(option.model or option.engine, usable,
                                                      option.compute_type or "this setting", option.vram_gb)
                )
                continue
        if engine.needs_gpu and machine.cuda_gpu is None and machine.metal_gpu is None:
            option.reason = "needs a GPU and none was found"
            rejected_ram = option
            continue
        if budget is not None and option.ram_gb > budget and not ignore_ram:
            option.reason = "needs about {0:.1f} GB of RAM{1}; {2}".format(
                option.ram_gb, " on the CPU" if option.device == "cpu" else "", budget_text
            )
            if rejected_ram is None or option.ram_gb < rejected_ram.ram_gb:
                rejected_ram = option
            continue
        feasible.append(option)
    if feasible:
        best = min(feasible, key=lambda o: (o.realtime_factor, -o.quality))
        if best.device == "cpu" and gpu_notes and engine.can_use_gpu:
            best.notes.append(gpu_notes[0] + ", so it runs on the CPU")
        return best, None
    if rejected_ram is not None:
        return None, rejected_ram
    fallback = variants[0]
    fallback.reason = gpu_notes[0] if gpu_notes else "cannot run on this machine"
    return None, fallback


def _order(candidates: List[Option], prefer: str) -> Tuple[List[Option], str]:
    def tie(o: Option) -> Tuple[str, str, str]:
        return (o.engine, o.model or "", o.device)

    def fastest(o: Option) -> Tuple[Any, ...]:
        return (not o.meets_floor, round(o.realtime_factor, 5), -round(o.quality, 4)) + tie(o)

    def best(o: Option) -> Tuple[Any, ...]:
        return (-round(o.quality, 4), round(o.realtime_factor, 5)) + tie(o)

    if prefer == "fast":
        ordered = sorted(candidates, key=fastest)
        return ordered, "fast" if ordered[0].meets_floor else "below-floor"
    if prefer == "accurate":
        ordered = sorted(candidates, key=best)
        return ordered, "accurate"
    within = [o for o in candidates if o.meets_floor and o.realtime_factor <= BALANCED_MAX_RTF]
    if within:
        rest = [o for o in candidates if o not in within]
        return sorted(within, key=best) + sorted(rest, key=fastest), "balanced"
    ordered = sorted(candidates, key=fastest)
    return ordered, "slow" if ordered[0].meets_floor else "below-floor"


# --------------------------------------------------------------------------- plan


def check_prefer(prefer: str) -> str:
    """Validate ``prefer`` and return it lower-cased."""
    if not isinstance(prefer, str) or prefer.strip().lower() not in PREFERENCES:
        raise ValueError("prefer must be one of {0}, got {1!r}".format(", ".join(PREFERENCES), prefer))
    return prefer.strip().lower()


def _model_matches(model: Optional[LocalModel], wanted: str) -> bool:
    if model is None:
        return False
    wanted_l = wanted.strip().lower()
    if model.name.lower() == wanted_l:
        return True
    if model.path:
        try:
            if os.path.normcase(os.path.abspath(wanted)) == os.path.normcase(os.path.abspath(model.path)):
                return True
        except (ValueError, OSError):
            return False
    return os.path.basename(model.path or "").lower() in (wanted_l, "ggml-{0}.bin".format(wanted_l), wanted_l + ".pt")


def _hardware(option: Option, machine: Machine) -> str:
    if option.device == "cuda" and machine.cuda_gpu is not None:
        return "the {0}".format(machine.cuda_gpu.name)
    if option.device == "metal":
        return "this Mac's GPU"
    if option.device == "gpu":
        return "this machine's GPU"
    return "this {0}-core CPU".format(machine.cpu_physical)


def _reason(option: Option, mode: str, machine: Machine, language: Optional[str], ram_text: str, forced: bool) -> str:
    thing = "the {0} model".format(option.model) if option.model else option.engine
    where = "you asked for" if forced else "on disk"
    hardware = _hardware(option, machine)
    if forced:
        text = "{0} is the model you asked for; on {1} it should run {2}".format(
            thing, hardware, option.speed.replace("about ", "at about ", 1))
    elif mode == "fast":
        text = "{0} is the fastest option {1} that is good enough for {2} ({3} on {4})".format(
            thing, where, language_name(language), option.speed, hardware)
    elif mode == "balanced":
        text = ("{0} is the most accurate option {1} that should still run at least 2x faster "
                "than real time on {2} ({3})").format(thing, where, hardware, option.speed)
    elif mode == "slow":
        text = ("nothing {0} is expected to reach 2x real time on {1}, so {2} was picked as the "
                "fastest ({3})").format(where, hardware, thing, option.speed)
    elif mode == "accurate":
        text = ("{0} is the most accurate option {1} that fits in memory; accuracy was put ahead of "
                "speed ({2} on {3})").format(thing, where, option.speed, hardware)
    else:
        text = "{0} is the best of what is {1} ({2} on {3})".format(thing, where, option.speed, hardware)
    text = text[0].upper() + text[1:] + "."
    if option.model and option.engine in WHISPER_ENGINES:
        tier = whisper_tier(language)
        if language == "en" and option.model.endswith(".en"):
            text += " English-only models are more accurate for English at this size."
        elif tier in ("moderate", "limited"):
            text += (" Whisper saw far less {0} than English in training, so models smaller than {1} "
                     "are not counted as good enough.").format(language_name(language), floor_name(language))
    if not option.meets_floor:
        text += (" It is smaller than {0} needs ({1} or larger), so expect a rough transcript."
                 ).format(language_name(language), floor_name(language))
    text += " It needs about {0:.1f} GB of RAM; {1}.".format(option.ram_gb, ram_text)
    return text


def _ram_phrase(budget: Optional[float], budget_text: str) -> str:
    if budget is None:
        return "the RAM could not be read, so memory was not checked"
    if budget_text.startswith("you allowed"):
        return budget_text
    return budget_text.replace("only ", "", 1)


def _alt_reason(alt: Option, chosen: Option) -> str:
    bits = [alt.speed, "about {0:.1f} GB RAM".format(alt.ram_gb)]
    if alt.quality > chosen.quality + 0.005:
        bits.append("more accurate")
    elif alt.quality < chosen.quality - 0.005:
        bits.append("less accurate")
    if not alt.meets_floor:
        bits.append("below the size this language needs")
    return ", ".join(bits)


def _hypothetical_engines(engines: Sequence[Engine], language: Optional[str]) -> List[Engine]:
    """The built-in engines as if installed, with every model they could fetch."""
    by_name = {e.name: e for e in engines}
    out: List[Engine] = []
    cuda_platform = sys.platform == "win32" or sys.platform.startswith("linux")
    for name in (FASTER_WHISPER, WHISPER_CPP, OPENAI_WHISPER, VOSK):
        real = by_name.get(name)
        if real is not None and real.failing:
            continue
        models = list(real.models_on_disk) if real is not None else []
        if name == VOSK:
            extra = vosk_catalog_model(language)
            catalog = [extra] if extra is not None and not any(m.supports(language) for m in models) else []
        else:
            catalog = catalog_models(name)
            present = {(m.family, m.multilingual) for m in models}
            catalog = [c for c in catalog if (c.family, c.multilingual) not in present]
        if real is not None and real.installed:
            gpu = real.can_use_gpu
        elif name == WHISPER_CPP:
            gpu = real.can_use_gpu if real is not None else False
        else:
            gpu = cuda_platform and name != VOSK
        out.append(Engine(name=name, installed=True, models_on_disk=models + catalog,
                          can_use_gpu=gpu, status=READY))
    for engine in engines:
        if engine.kind == "custom" and engine.ready:
            out.append(engine)
    return out


def _fetch_hint(option: Option, language: Optional[str]) -> str:
    model = option.model or ""
    if option.engine == FASTER_WHISPER:
        return "fetch the {0} model: python -c \"from faster_whisper import WhisperModel; WhisperModel('{0}')\"".format(model)
    if option.engine == OPENAI_WHISPER:
        return "fetch the {0} model: python -c \"import whisper; whisper.load_model('{0}')\"".format(model)
    if option.engine == WHISPER_CPP:
        folder = _short(whisper_cpp_dirs(None)[0])
        return ("download ggml-{0}.bin from https://huggingface.co/ggerganov/whisper.cpp into {1} "
                "(or any folder named in WHISPER_CPP_MODELS)").format(model, folder)
    if option.engine == VOSK:
        return ("download {0} from https://alphacephei.com/vosk/models and unzip it into "
                "~/.cache/vosk").format(model)
    return "put the {0} model on disk".format(model)


def _better(candidate: Option, current: Optional[Option], prefer: str) -> bool:
    if current is None:
        return True
    if (candidate.engine, candidate.model, candidate.device) == (current.engine, current.model, current.device):
        return False
    if candidate.meets_floor and not current.meets_floor:
        return True
    if prefer == "fast":
        return candidate.meets_floor and candidate.realtime_factor <= 0.6 * current.realtime_factor
    if prefer == "balanced":
        quick_enough = candidate.realtime_factor <= max(BALANCED_MAX_RTF, current.realtime_factor)
        return candidate.quality >= current.quality + 0.04 and quick_enough
    return candidate.quality >= current.quality + 0.04


def _suggestions(
    engines: Sequence[Engine],
    machine: Machine,
    language: Optional[str],
    prefer: str,
    budget: Optional[float],
    budget_text: str,
    chosen: Optional[Option],
    only_engine: Optional[str] = None,
) -> List[str]:
    by_name = {e.name: e for e in engines}
    out: List[str] = []
    for engine in engines:
        if engine.failing and only_engine in (None, engine.name.lower()):
            out.append("repair {0}: {1}".format(engine.name, engine.error))
    pool: List[Option] = []
    for engine in _hypothetical_engines(engines, language):
        if only_engine is not None and engine.name.lower() != only_engine:
            continue
        models: List[Optional[LocalModel]] = list(engine.models_on_disk) if engine.needs_model else [None]
        for model in models:
            option, _rejected = _evaluate(engine, model, machine, language, budget, budget_text)
            if option is not None:
                pool.append(option)
    if not pool:
        if chosen is None and only_engine is None:
            out.append(
                "no engine known to this package handles {0} within the memory available; "
                "a custom engine can be added with register()".format(language_name(language))
            )
        return out
    ordered, _mode = _order(pool, prefer)
    used_engines = set()
    for option in ordered:
        if len(out) >= 3 or len(used_engines) >= 2:
            break
        if option.engine in used_engines or not _better(option, chosen, prefer):
            continue
        real = by_name.get(option.engine)
        installed = real is not None and real.installed and not real.failing
        if option.engine not in INSTALL_HINTS:
            continue
        if installed and option.on_disk:
            continue
        if not installed:
            text = INSTALL_HINTS[option.engine]
            if option.on_disk:
                text += ", and it will use the {0} model already on disk".format(option.model)
            else:
                text += ", then " + _fetch_hint(option, language)
                existing = [m.name for m in (real.models_on_disk if real is not None else []) if m.on_disk]
                if existing:
                    text += "; it will also find {0} already on disk".format(", ".join(existing[:4]))
        else:
            text = "for {0}, {1}".format(option.engine, _fetch_hint(option, language))
        if chosen is None:
            benefit = "{0} on {1}, {2}, about {3:.1f} GB of RAM".format(
                option.model or option.engine, _hardware(option, machine), option.speed, option.ram_gb)
        elif prefer == "fast" and option.realtime_factor < chosen.realtime_factor:
            benefit = "about {0:.0f}x faster than {1}".format(
                chosen.realtime_factor / max(option.realtime_factor, 1e-9), chosen.model or chosen.engine)
        else:
            benefit = "more accurate than {0} for {1}, {2}".format(
                chosen.model or chosen.engine, language_name(language), option.speed)
        out.append("{0} ({1})".format(text, benefit))
        used_engines.add(option.engine)
    return out


def _nothing_reason(engines: Sequence[Engine], rejected: Sequence[Option], language: Optional[str], only_engine: Optional[str]) -> str:
    installed = [e for e in engines if e.installed]
    if only_engine is not None:
        match = [e for e in engines if e.name == only_engine]
        if not match:
            return "no engine called {0!r} is known (built-in: faster-whisper, openai-whisper, whisper.cpp, vosk)".format(only_engine)
        engine = match[0]
        if not engine.ready and not rejected:
            return "{0} is not usable: {1}".format(engine.name, engine.describe().split(" - ", 1)[-1])
    if not installed:
        names = ", ".join(e.name for e in engines)
        return "none of the engines this package knows ({0}) is installed.".format(names)
    parts: List[str] = []
    for engine in installed:
        if engine.status == "no-model":
            parts.append("{0} is installed but has no model on disk".format(engine.name))
        elif engine.failing:
            parts.append("{0} is installed but failing ({1})".format(engine.name, engine.error))
        elif not engine.runnable:
            parts.append("{0} has no transcribe function registered".format(engine.name))
    for option in rejected[:4]:
        parts.append("{0}: {1}".format(option.label() if option.model else option.engine, option.reason))
    if not parts:
        return "nothing installed can transcribe {0}.".format(language_name(language))
    return "nothing installed can transcribe {0} here - {1}.".format(language_name(language), "; ".join(parts))


def plan(
    engines: Sequence[Engine],
    machine: Machine,
    *,
    language: Optional[str] = "en",
    prefer: str = "balanced",
    max_ram_gb: Optional[float] = None,
    runnable_only: bool = False,
    only_engine: Optional[str] = None,
    only_model: Optional[str] = None,
    extra_models: Sequence[LocalModel] = (),
) -> Choice:
    """Pick an engine, a model and a device. Never raises for "nothing installed"."""
    code = normalize_language(language)
    prefer = check_prefer(prefer)
    budget, budget_text = ram_budget(machine, max_ram_gb)
    wanted_engine = only_engine.strip().lower() if isinstance(only_engine, str) else None
    candidates: List[Option] = []
    rejected: List[Option] = []
    notes: List[str] = []
    if budget is None:
        notes.append("the RAM could not be read, so memory limits were not checked")
    for engine in engines:
        if not engine.ready:
            continue
        if runnable_only and not engine.runnable:
            continue
        if wanted_engine is not None and engine.name.lower() != wanted_engine:
            continue
        models: List[Optional[LocalModel]] = list(engine.models_on_disk) if engine.needs_model else [None]
        models += [m for m in extra_models if m.engine == engine.name]
        if only_model is not None and engine.needs_model:
            models = [m for m in models if _model_matches(m, only_model)]
        for model in models:
            option, reject = _evaluate(engine, model, machine, code, budget, budget_text,
                                       ignore_ram=only_model is not None)
            if option is not None:
                candidates.append(option)
            elif reject is not None:
                rejected.append(reject)
    if only_model is not None:
        candidates_named = [c for c in candidates if c.model is None or _model_matches_option(c, only_model)]
        candidates = candidates_named or candidates
        if not candidates and not rejected:
            notes.append("no model called {0!r} is on disk for the engines here".format(only_model))

    choice = Choice(engine=None, model=None, device=None, reason="", language=code, prefer=prefer,
                    rejected=rejected, notes=notes, ram_budget_gb=budget, machine=machine,
                    engines=list(engines))
    if not candidates:
        reason = _nothing_reason(engines, rejected, code, wanted_engine)
        choice.reason = reason[0].upper() + reason[1:]
        if not choice.reason.endswith("."):
            choice.reason += "."
        choice.missing = _suggestions(engines, machine, code, prefer, budget, budget_text, None, wanted_engine)
        return choice

    ordered, mode = _order(candidates, prefer)
    pick = ordered[0]
    ram_text = _ram_phrase(budget, budget_text)
    forced = only_model is not None
    choice.engine = pick.engine
    choice.model = pick.model
    choice.device = pick.device
    choice.compute_type = pick.compute_type
    choice.model_path = pick.model_path
    choice.estimated_ram_gb = pick.ram_gb
    choice.estimated_vram_gb = pick.vram_gb
    choice.realtime_factor = pick.realtime_factor
    choice.option = pick
    choice.reason = _reason(pick, mode, machine, code, ram_text, forced)
    choice.notes.extend(pick.notes)
    alternatives: List[Option] = []
    seen = {(pick.engine, pick.model, pick.model_path)}
    for alt in ordered[1:]:
        key = (alt.engine, alt.model, alt.model_path)
        if key in seen:
            continue
        seen.add(key)
        alt.reason = _alt_reason(alt, pick)
        alternatives.append(alt)
        if len(alternatives) >= 5:
            break
    choice.alternatives = alternatives
    choice.missing = _suggestions(engines, machine, code, prefer, budget, budget_text, pick, wanted_engine)
    return choice


def _model_matches_option(option: Option, wanted: str) -> bool:
    model = LocalModel(name=option.model or "", engine=option.engine, family=option.family or "",
                       path=option.model_path or "")
    return _model_matches(model, wanted)
