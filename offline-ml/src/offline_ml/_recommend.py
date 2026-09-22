"""Pick the model that will actually fit and run on this machine.

Everything here is arithmetic on the numbers :func:`offline_ml.detect` reported.
No network, no downloads, no GPU library, no surprises.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ._hardware import Hardware, detect
from ._models import ModelSpec

#: The values ``prefer=`` accepts.
PREFERENCES = ("balanced", "quality", "speed", "smallest")

_PREFER_PHRASE = {
    "balanced": "the best balance of quality and speed among the {n} models that fit",
    "quality": "the highest-quality of the {n} models that fit",
    "speed": "the fastest of the {n} models that fit",
    "smallest": "the smallest of the {n} models that fit",
}
_MAX_HEADROOM = 10.0

#: What to suggest next, per kind of rejection. Keyed by :attr:`_Rejection.kind`.
_ADVICE = {
    "task": "Pass a different task, or leave task out to consider all of them.",
    "ram": "Look for a smaller or more heavily quantized build.",
    "disk": "Free up disk space, or look for a smaller build.",
}


def _article(word: Optional[str]) -> str:
    """``"a"`` or ``"an"`` for the word that follows. Spelling, not phonetics."""
    for char in str(word or ""):
        if char.isalpha():
            return "an" if char.lower() in "aeiou" else "a"
        if char.isdigit():
            return "a"
    return "a"


@dataclass(frozen=True)
class _Rejection:
    """Why one model was ruled out.

    Attributes:
        kind: ``"task"``, ``"ram"`` or ``"disk"`` - which gate it failed, so the
            advice that follows can match the actual problem.
        text: The clause, with no subject of its own, so that
            ``f"'{name}' {text}."`` reads as one sentence.
    """

    kind: str
    text: str


@dataclass
class _Candidate:
    """A model that fits, with the numbers that decided it."""

    spec: ModelSpec
    ram_gb: float
    vram_gb: float
    disk_gb: float
    device: str
    tight: bool
    disk_unknown: bool = False
    score: float = 0.0


@dataclass
class Recommendation:
    """What to run, where, and why.

    Attributes:
        model: The model to use, or ``None`` when nothing fits.
        device: ``"cuda"``, ``"mps"`` or ``"cpu"`` - where to load it.
        reason: Plain language, safe to show a user as-is.
        alternatives: The other models that fit, best first.
        rejected: ``{name: why it will not fit}`` for everything ruled out.
        fits: True when :attr:`model` is not None.
        requirements: ``{name: {"disk_gb", "ram_gb", "vram_gb"}}`` for every
            model considered, in GiB, headroom already included.
        hardware: The machine this was decided against.
    """

    model: Optional[ModelSpec]
    device: str
    reason: str
    alternatives: List[ModelSpec] = field(default_factory=list)
    rejected: Dict[str, str] = field(default_factory=dict)
    fits: bool = False
    requirements: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hardware: Optional[Hardware] = None
    task: Optional[str] = None
    prefer: str = "balanced"
    headroom: float = 0.2

    @property
    def name(self) -> Optional[str]:
        """The chosen model's name, or None."""
        return self.model.name if self.model is not None else None

    def summary(self, limit: int = 5) -> str:
        """The human-readable report. Plain ASCII punctuation only.

        Args:
            limit: how many alternatives and rejections to list. It is a count,
                so anything below zero is treated as zero; the "and N more"
                line always names the number actually withheld.
        """
        limit = max(0, int(limit))
        if self.model is None:
            head = f"offline-ml: nothing fits; this machine would use {self.device}"
        else:
            head = f"offline-ml: run '{self.model.name}' on {self.device}"
        lines = [head]
        wrapped = textwrap.wrap(self.reason, width=62) or [""]
        for index, chunk in enumerate(wrapped):
            label = "  reason    : " if index == 0 else "              "
            lines.append(label + chunk)
        if self.model is not None:
            need = self.requirements.get(self.model.name, {})
            lines.append(
                "  needs     : "
                f"{need.get('disk_gb', 0.0):.1f} GiB on disk, "
                f"{need.get('ram_gb', 0.0):.1f} GiB of RAM, "
                f"{need.get('vram_gb', 0.0):.1f} GiB of VRAM to use a GPU"
            )
        if self.hardware is not None:
            lines.append(f"  machine   : {self.hardware.short()}")
        lines.append(f"  asked for : prefer={self.prefer}, headroom={self.headroom:g}"
                     + (f", task={self.task}" if self.task else ""))
        if self.alternatives:
            shown = self.alternatives[:limit]
            lines.append("  alternatives:")
            for spec in shown:
                lines.append(f"    {spec}")
            extra = len(self.alternatives) - len(shown)
            if extra > 0:
                lines.append(f"    and {extra} more")
        if self.rejected:
            entries = list(self.rejected.items())
            shown_entries = entries[:limit]
            lines.append("  rejected:")
            for name, why in shown_entries:
                lines.append(f"    {name}: {why}")
            extra = len(entries) - len(shown_entries)
            if extra > 0:
                lines.append(f"    and {extra} more")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of everything above."""
        return {
            "fits": self.fits,
            "model": self.model.to_dict() if self.model is not None else None,
            "device": self.device,
            "reason": self.reason,
            "alternatives": [spec.to_dict() for spec in self.alternatives],
            "rejected": dict(self.rejected),
            "requirements": {
                name: dict(values) for name, values in self.requirements.items()
            },
            "task": self.task,
            "prefer": self.prefer,
            "headroom": self.headroom,
            "hardware": self.hardware.to_dict() if self.hardware is not None else None,
        }

    def __str__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------- #
# input checking
# --------------------------------------------------------------------------- #
def _check_prefer(prefer: str) -> str:
    if not isinstance(prefer, str) or prefer.lower() not in PREFERENCES:
        raise ValueError(
            f"prefer must be one of {', '.join(PREFERENCES)}, got {prefer!r}"
        )
    return prefer.lower()


def _check_headroom(headroom: float) -> float:
    if isinstance(headroom, bool) or not isinstance(headroom, (int, float)):
        raise ValueError(
            f"headroom must be a number, got {type(headroom).__name__}"
        )
    value = float(headroom)
    if not 0.0 <= value <= _MAX_HEADROOM:
        raise ValueError(
            "headroom must be between 0 and "
            f"{_MAX_HEADROOM:g} (0.2 means 'leave 20% spare'), got {value:g}"
        )
    return value


def _as_specs(models: Iterable[Any]) -> List[ModelSpec]:
    if isinstance(models, (str, bytes)) or isinstance(models, dict):
        raise ValueError(
            "models must be a list of ModelSpec or dicts, got "
            f"{type(models).__name__}. Pass [{type(models).__name__}] to check one model."
        )
    try:
        items = list(models)
    except TypeError as exc:
        raise ValueError(
            f"models must be a list of ModelSpec or dicts, got {type(models).__name__}"
        ) from exc
    if not items:
        raise ValueError(
            "no models to choose from: pass a non-empty list of ModelSpec or "
            "dicts with at least 'name' and 'size_gb'"
        )
    specs = [ModelSpec.coerce(item) for item in items]
    seen: Dict[str, int] = {}
    for spec in specs:
        seen[spec.name] = seen.get(spec.name, 0) + 1
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            "duplicate model names: " + ", ".join(repr(name) for name in duplicates)
            + ". Every model needs its own name."
        )
    return specs


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _needs(spec: ModelSpec, headroom: float) -> Tuple[float, float, float]:
    """(ram_gb, vram_gb, disk_gb) this model needs, headroom included."""
    padded = spec.size_gb * (1.0 + headroom)
    ram = max(spec.min_ram_gb or 0.0, padded)
    vram = max(spec.min_vram_gb or 0.0, padded)
    return round(ram, 2), round(vram, 2), round(spec.size_gb, 2)


def _evaluate(
    spec: ModelSpec, hardware: Hardware, headroom: float, task: Optional[str]
) -> Tuple[Optional[_Candidate], Optional[_Rejection]]:
    """Return (candidate, None) when the model fits, or (None, why not).

    Every rejection clause is written without a subject of its own, so that
    ``f"'{name}' {clause}."`` reads as one sentence wherever it is spliced in.
    """
    ram, vram, disk = _needs(spec, headroom)
    if task and spec.task and spec.task.lower() != task.lower():
        return None, _Rejection(
            "task",
            f"is {_article(spec.task)} {spec.task!r} model and you asked "
            f"for {task!r}",
        )
    if ram > hardware.ram_total_gb:
        return None, _Rejection(
            "ram",
            f"needs about {ram:.1f} GiB of RAM, this machine has only "
            f"{hardware.ram_total_gb:.1f} GiB in total",
        )
    free_disk = hardware.disk_free_gb
    # None means "could not be read", which is missing information, not proof
    # of a full disk: skip the gate and say so in the reason instead.
    if free_disk is not None and disk > free_disk:
        return None, _Rejection(
            "disk",
            f"needs {disk:.1f} GiB on disk, only {free_disk:.1f} GiB "
            f"is free on {hardware.disk_path}",
        )
    device = "cpu"
    if hardware.has_gpu and vram <= hardware.vram_gb:
        device = hardware.device
    return (
        _Candidate(
            spec=spec,
            ram_gb=ram,
            vram_gb=vram,
            disk_gb=disk,
            device=device,
            tight=ram > hardware.ram_available_gb,
            disk_unknown=free_disk is None,
        ),
        None,
    )


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #
def _minmax(values: Sequence[float]) -> List[float]:
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _trait(
    candidates: Sequence[_Candidate], attribute: str, fallback: Sequence[float]
) -> List[float]:
    """Normalise a declared trait to 0..1, standing in size where it is missing."""
    declared = [getattr(candidate.spec, attribute) for candidate in candidates]
    known = [value for value in declared if value is not None]
    if not known:
        return list(fallback)
    low, high = min(known), max(known)
    span = high - low
    out: List[float] = []
    for value, stand_in in zip(declared, fallback):
        if value is None:
            out.append(stand_in)
        elif span <= 1e-12:
            out.append(0.5)
        else:
            out.append((value - low) / span)
    return out


def _rank(candidates: List[_Candidate], prefer: str) -> List[_Candidate]:
    """Best first, with ties broken by size then name so it is deterministic."""
    by_size = _minmax([candidate.spec.size_gb for candidate in candidates])
    quality = _trait(candidates, "quality", by_size)
    speed = _trait(candidates, "speed", [1.0 - value for value in by_size])
    for candidate, q, s in zip(candidates, quality, speed):
        accel = 1.0 if candidate.device != "cpu" else 0.0
        tight = 1.0 if candidate.tight else 0.0
        if prefer == "quality":
            candidate.score = q + 0.05 * accel - 0.10 * tight
        elif prefer == "speed":
            candidate.score = s + 0.10 * accel - 0.10 * tight
        elif prefer == "smallest":
            candidate.score = -candidate.spec.size_gb
        else:  # balanced: quality leans a little ahead of speed, comfort matters
            candidate.score = 0.50 * q + 0.40 * s + 0.10 * accel - 0.15 * tight
    return sorted(
        candidates, key=lambda c: (-c.score, c.spec.size_gb, c.spec.name)
    )


# --------------------------------------------------------------------------- #
# explaining
# --------------------------------------------------------------------------- #
def _device_sentence(candidate: _Candidate, hardware: Hardware) -> str:
    gpu = hardware.gpu
    if candidate.device != "cpu":
        return (
            f"Its {candidate.vram_gb:.1f} GiB of VRAM fits the {hardware.vram_gb:.1f} "
            f"GiB on {gpu.name}, so it will run on the GPU ({candidate.device})."
        )
    if gpu is None:
        if hardware.gpus:
            return (
                f"The only graphics device found ({hardware.gpus[0].name}) has no "
                "usable compute backend, so it will run on the CPU."
            )
        return "No GPU was found, so it will run on the CPU."
    if gpu.vram_gb is None:
        return (
            f"{gpu.name} was found but its memory could not be read, so it will "
            "run on the CPU."
        )
    return (
        f"It would need {candidate.vram_gb:.1f} GiB of VRAM and {gpu.name} has only "
        f"{hardware.vram_gb:.1f} GiB, so it will run on the CPU."
    )


def _ram_sentence(candidate: _Candidate, hardware: Hardware) -> str:
    text = (
        f"It needs about {candidate.ram_gb:.1f} GiB of RAM out of "
        f"{hardware.ram_total_gb:.1f} GiB"
    )
    if candidate.tight:
        return (
            text + f", but only {hardware.ram_available_gb:.1f} GiB is free right now, "
            "so close something before loading it."
        )
    return text + f", {hardware.ram_available_gb:.1f} GiB of it free right now."


def _disk_sentence(candidate: _Candidate, hardware: Hardware) -> str:
    """Said only when the free-space check could not be run at all."""
    if not candidate.disk_unknown:
        return ""
    return (
        f"Free space on {hardware.disk_path} could not be read, so the disk "
        f"check was skipped; make sure there is room for {candidate.disk_gb:.1f} "
        "GiB of weights."
    )


def _why_chosen(
    candidate: _Candidate, candidates: List[_Candidate], hardware: Hardware, prefer: str
) -> str:
    if len(candidates) == 1:
        opening = f"'{candidate.spec.name}' is the only model that fits this machine."
    else:
        phrase = _PREFER_PHRASE[prefer].format(n=len(candidates))
        opening = f"'{candidate.spec.name}' is {phrase}."
    parts = [
        opening,
        _ram_sentence(candidate, hardware),
        _device_sentence(candidate, hardware),
        _disk_sentence(candidate, hardware),
    ]
    return " ".join(part for part in parts if part)


def _why_nothing(
    specs: List[ModelSpec],
    rejections: Dict[str, _Rejection],
    hardware: Hardware,
    task: Optional[str],
) -> str:
    count = len(specs)
    plural = "model" if count == 1 else "models"
    if task and rejections and all(why.kind == "task" for why in rejections.values()):
        return (
            f"None of the {count} {plural} is {_article(task)} {task!r} model. "
            + _ADVICE["task"]
        )
    smallest = min(specs, key=lambda spec: spec.size_gb)
    why = rejections.get(smallest.name)
    clause = why.text if why is not None else "does not fit"
    advice = _ADVICE.get(why.kind, _ADVICE["ram"]) if why is not None else _ADVICE["ram"]
    return (
        f"None of the {count} {plural} fits this machine ({hardware.short()}). "
        f"The smallest one, '{smallest.name}', {clause}. {advice}"
    )


# --------------------------------------------------------------------------- #
# the public API
# --------------------------------------------------------------------------- #
def recommend(
    models: Iterable[Any],
    *,
    task: Optional[str] = None,
    prefer: str = "balanced",
    headroom: float = 0.2,
    hardware: Optional[Hardware] = None,
) -> Recommendation:
    """Choose the model to run here, and say why.

    Args:
        models: A non-empty list of :class:`ModelSpec`, or of dicts with the
            same keys. At minimum each needs ``name`` and ``size_gb``.
        task: Only consider models for this task. A model that declares no
            task is considered for every task.
        prefer: ``"balanced"`` (default), ``"quality"``, ``"speed"`` or
            ``"smallest"``.
        headroom: Spare room to leave, as a fraction. ``0.2`` means a 4 GiB
            model is treated as needing 4.8 GiB. A model's own ``min_ram_gb``
            or ``min_vram_gb`` wins when it is larger.
        hardware: The machine to decide against. Defaults to
            :func:`~offline_ml.detect`; pass one in to plan for another box.

    Returns:
        A :class:`Recommendation`. Check ``.fits`` before using ``.model``:
        when nothing fits, ``.model`` is None and ``.rejected`` says why for
        every candidate.

    Raises:
        ValueError: if ``models`` is empty, if two models share a name, or if
            ``prefer`` / ``headroom`` / a model spec is not valid.
    """
    specs = _as_specs(models)
    prefer = _check_prefer(prefer)
    headroom = _check_headroom(headroom)
    machine = hardware if hardware is not None else detect()

    requirements: Dict[str, Dict[str, float]] = {}
    candidates: List[_Candidate] = []
    rejections: Dict[str, _Rejection] = {}
    for spec in specs:
        ram, vram, disk = _needs(spec, headroom)
        requirements[spec.name] = {"disk_gb": disk, "ram_gb": ram, "vram_gb": vram}
        candidate, why = _evaluate(spec, machine, headroom, task)
        if candidate is None:
            rejections[spec.name] = why or _Rejection("ram", "does not fit")
        else:
            candidates.append(candidate)
    rejected = {name: why.text for name, why in rejections.items()}

    if not candidates:
        return Recommendation(
            model=None,
            device=machine.device,
            reason=_why_nothing(specs, rejections, machine, task),
            alternatives=[],
            rejected=rejected,
            fits=False,
            requirements=requirements,
            hardware=machine,
            task=task,
            prefer=prefer,
            headroom=headroom,
        )

    ordered = _rank(candidates, prefer)
    winner = ordered[0]
    return Recommendation(
        model=winner.spec,
        device=winner.device,
        reason=_why_chosen(winner, ordered, machine, prefer),
        alternatives=[candidate.spec for candidate in ordered[1:]],
        rejected=rejected,
        fits=True,
        requirements=requirements,
        hardware=machine,
        task=task,
        prefer=prefer,
        headroom=headroom,
    )


def fits(
    spec: Any, *, headroom: float = 0.2, hardware: Optional[Hardware] = None
) -> bool:
    """True when this model can run on this machine at all.

    Args:
        spec: A :class:`ModelSpec`, a dict of the same shape, or just a size in
            GiB, e.g. ``fits(7.5)``.
        headroom: Same meaning as in :func:`recommend`.
        hardware: Defaults to :func:`~offline_ml.detect`.

    Returns:
        True when the model fits the installed RAM and the free disk space.
        It may still be tight if other programs are using the RAM right now;
        :func:`recommend` says so in its reason.
    """
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        model = ModelSpec(name="model", size_gb=float(spec))
    else:
        model = ModelSpec.coerce(spec)
    machine = hardware if hardware is not None else detect()
    candidate, _ = _evaluate(model, machine, _check_headroom(headroom), None)
    return candidate is not None
