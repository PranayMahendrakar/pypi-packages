"""``ModelSpec``: what you tell offline-ml about a model you are considering."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional

_FIELD_NAMES = (
    "name",
    "size_gb",
    "min_ram_gb",
    "min_vram_gb",
    "task",
    "quality",
    "speed",
)


def _number(
    value: Any,
    label: str,
    model: str,
    *,
    positive: bool = False,
    signed: bool = False,
) -> float:
    """Check one number on a model spec.

    Args:
        positive: the value must be greater than 0 (a size).
        signed: any finite number is allowed, negatives included. Used for the
            ``quality`` and ``speed`` ratings, which are on whatever scale you
            like, so a centred scale such as -5..+5 is legitimate.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"model {model!r}: {label} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"model {model!r}: {label} must be a finite number, got {value!r}")
    if positive and number <= 0:
        raise ValueError(f"model {model!r}: {label} must be greater than 0, got {number:g}")
    if not positive and not signed and number < 0:
        raise ValueError(f"model {model!r}: {label} cannot be negative, got {number:g}")
    return number


@dataclass(frozen=True)
class ModelSpec:
    """One model you might run, described well enough to judge whether it fits.

    Attributes:
        name: How you refer to the model. Must be unique within one call.
        size_gb: What the weights take on disk, in GiB. Required.
        min_ram_gb: RAM the model needs, in GiB, if you know better than
            "its size plus headroom".
        min_vram_gb: VRAM the model needs, in GiB, same idea.
        task: What it is for, e.g. ``"chat"``, ``"asr"``, ``"embedding"``. A
            model with no task matches whatever task you ask for.
        quality: How good it is, on any numeric scale you like, higher is
            better. Negative values are fine, so a centred scale such as
            -5..+5 works. Optional; size stands in for it when it is missing.
        speed: How fast it is, same idea. Optional.
    """

    name: str
    size_gb: float
    min_ram_gb: Optional[float] = None
    min_vram_gb: Optional[float] = None
    task: Optional[str] = None
    quality: Optional[float] = None
    speed: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("every model needs a non-empty name")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self, "size_gb", _number(self.size_gb, "size_gb", self.name, positive=True)
        )
        for label in ("min_ram_gb", "min_vram_gb"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, _number(value, label, self.name))
        # quality and speed are ratings on the caller's own scale, so a
        # negative value is meaningful rather than a mistake.
        for label in ("quality", "speed"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(
                    self, label, _number(value, label, self.name, signed=True)
                )
        if self.task is not None:
            if not isinstance(self.task, str):
                raise ValueError(
                    f"model {self.name!r}: task must be a string or None, "
                    f"got {type(self.task).__name__}"
                )
            object.__setattr__(self, "task", self.task.strip() or None)

    @classmethod
    def coerce(cls, obj: Any) -> "ModelSpec":
        """Accept a ModelSpec or a dict of the same shape; reject anything else."""
        if isinstance(obj, ModelSpec):
            return obj
        if isinstance(obj, Mapping):
            data = dict(obj)
            unknown = sorted(set(data) - set(_FIELD_NAMES))
            if unknown:
                raise ValueError(
                    "unknown model keys: "
                    + ", ".join(repr(key) for key in unknown)
                    + ". Allowed keys are: "
                    + ", ".join(_FIELD_NAMES)
                )
            if "name" not in data:
                raise ValueError(f"model dict is missing 'name': {obj!r}")
            if "size_gb" not in data:
                raise ValueError(
                    f"model {data['name']!r} is missing 'size_gb' "
                    "(the size of the weights on disk, in GiB)"
                )
            return cls(**data)
        raise ValueError(
            "a model must be a ModelSpec or a dict with at least 'name' and "
            f"'size_gb', got {type(obj).__name__}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict, dropping the fields you did not set."""
        out: Dict[str, Any] = {}
        for spec_field in fields(self):
            value = getattr(self, spec_field.name)
            if value is not None:
                out[spec_field.name] = value
        return out

    def __str__(self) -> str:
        return f"{self.name} ({self.size_gb:.1f} GiB)"
