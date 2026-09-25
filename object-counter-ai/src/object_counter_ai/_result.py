"""The result of one count, and how it explains itself."""
from __future__ import annotations

from collections import Counter as _Tally
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

METHOD_DETECTOR = "detector"
METHOD_CLASSICAL = "classical"


def _num(value: Any, places: int = 2) -> Any:
    """JSON-safe number: ints stay ints, floats are rounded."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    if value.is_integer():
        return int(value)
    return round(value, places)


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating)) or value is None:
        return _num(value, 3)
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    return str(value)


def _fmt(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


@dataclass
class CountResult:
    """How many things were counted in one image, and why the number is what it is.

    ``boxes`` are ``(left, top, right, bottom)`` in pixels with right and bottom
    exclusive. ``labels``, ``scores``, ``areas``, ``centroids`` and (inside a
    :class:`Counter`) ``track_ids`` line up with ``boxes``. ``confidence`` is in
    0-1: for a detector it is the mean score of the kept boxes (None when the
    detector gives no scores); for the classical counter it is a heuristic score
    of how clean the picture looked, not a probability. ``notes`` says what
    lowered it and anything else worth knowing.
    """

    count: int
    boxes: List[Tuple[float, float, float, float]]
    labels: List[str]
    scores: List[Optional[float]]
    areas: List[float]
    centroids: List[Tuple[float, float]]
    confidence: Optional[float]
    method: str
    image_size: Tuple[int, int]
    image_kind: str = "grey"
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    region: Optional[Dict[str, Any]] = None
    frame: Optional[int] = None
    track_ids: List[int] = field(default_factory=list)
    crossings: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def by_label(self) -> Dict[str, int]:
        """Counts per label, largest first (ties alphabetical)."""
        tally = _Tally(self.labels)
        return dict(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def ok(self) -> bool:
        """False when the detector failed and this image was not really counted."""
        return self.error is None

    # ------------------------------------------------------------------ text
    def _noun(self) -> str:
        if self.method == METHOD_CLASSICAL:
            return "blob" if self.count == 1 else "blobs"
        return "object" if self.count == 1 else "objects"

    def summary(self) -> str:
        """Human-readable explanation, plain ASCII."""
        width, height = self.image_size
        where = f" (frame {self.frame})" if self.frame is not None else ""
        how = "classical blob counter" if self.method == METHOD_CLASSICAL else "your detector"
        lines: List[str] = []
        if self.error is not None:
            lines.append(f"object-counter-ai{where}: NOT counted - the detector failed")
            lines.append(f"  error       {self.error}")
            lines.append(f"  image       {width} x {height} pixels, {self.image_kind}")
            lines.append("  count       0 reported, but nothing was counted; treat this image as missing")
            for note in self.notes:
                lines.append(f"  note        {note}")
            return "\n".join(lines)

        lines.append(f"object-counter-ai{where}: {self.count} {self._noun()} counted ({how})")
        lines.append(f"  image       {width} x {height} pixels, {self.image_kind}")
        if self.region is not None:
            lines.append(f"  region      {self.details.get('region_text', 'set')}")
        if self.by_label:
            parts = ", ".join(f"{name} {n}" for name, n in self.by_label.items())
            lines.append(f"  by label    {parts}")
        if self.areas:
            lo, hi = min(self.areas), max(self.areas)
            med = float(np.median(self.areas))
            unit = "px" if self.method == METHOD_CLASSICAL else "px box"
            if lo == hi:
                lines.append(f"  sizes       {_fmt(lo)} {unit} each")
            else:
                lines.append(f"  sizes       {_fmt(lo)} to {_fmt(hi)} {unit}, median {_fmt(med)}")
        if self.confidence is None:
            lines.append("  confidence  unknown (the detector gave no scores)")
        elif self.method == METHOD_CLASSICAL:
            lines.append(f"  confidence  {self.confidence:.2f} (a heuristic score, not a probability)")
        else:
            lines.append(f"  confidence  {self.confidence:.2f} (mean detector score)")
        d = self.details
        if self.method == METHOD_CLASSICAL and "threshold" in d:
            bg = d.get("background", {})
            polarity = bg.get("polarity", "none")
            shade = {"light": "light, things darker", "dark": "dark, things lighter",
                     "colour": "things differ from it in colour"}.get(polarity, "no foreground found")
            model = bg.get("model", "flat")
            model_text = "even" if model == "flat" else f"uneven, fitted (varies by {_fmt(bg.get('variation', 0))})"
            lines.append(f"  background  {shade}; {model_text}")
            lines.append(
                f"  threshold   {_fmt(d['threshold'])} levels from the background "
                f"(noise {d.get('noise', 0):.1f}, floor {_fmt(d.get('floor', 0))})"
            )
            ignored = []
            if d.get("specks"):
                ignored.append(f"{d['specks']} under min_area {_fmt(d['min_area'])} px")
            if d.get("too_large"):
                ignored.append(f"{d['too_large']} over max_area {_fmt(d['max_area'])} px")
            if ignored:
                lines.append(f"  ignored     {'; '.join(ignored)}")
            if d.get("split_blobs"):
                lines.append(
                    f"  split       {d['split_blobs']} touching shape(s) cut at a narrow neck "
                    f"into {d['split_into']} blobs"
                )
        if self.method == METHOD_DETECTOR:
            dropped = []
            if d.get("clipped"):
                dropped.append(f"{d['clipped']} clipped to the image")
            if d.get("outside"):
                dropped.append(f"{d['outside']} outside the image dropped")
            if d.get("dropped_by_area"):
                dropped.append(f"{d['dropped_by_area']} dropped by min_area/max_area")
            if d.get("dropped_by_region"):
                dropped.append(f"{d['dropped_by_region']} centred outside the region")
            if dropped:
                lines.append(f"  boxes       {d.get('received', 0)} returned; " + ", ".join(dropped))
        if self.frame is not None and self.track_ids:
            lines.append(f"  tracks      {len(set(self.track_ids))} tracked in this frame")
        if self.crossings:
            parts = ", ".join(f"{name}: {n}" for name, n in self.crossings.items())
            lines.append(f"  crossed     {parts} (new this frame)")
        for note in self.notes:
            lines.append(f"  note        {note}")
        if self.method == METHOD_CLASSICAL:
            lines.append("  caveat      it counts high-contrast blobs, not objects; things touching")
            lines.append("              with no narrow neck between them count as one")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------ data
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of everything above."""
        width, height = self.image_size
        objects = []
        for i, box in enumerate(self.boxes):
            item: Dict[str, Any] = {
                "box": [_num(v, 2) for v in box],
                "label": self.labels[i],
                "score": _num(self.scores[i], 4) if i < len(self.scores) else None,
                "area": _num(self.areas[i], 2) if i < len(self.areas) else None,
                "centre": [_num(v, 2) for v in self.centroids[i]] if i < len(self.centroids) else None,
            }
            if self.track_ids:
                item["track_id"] = int(self.track_ids[i])
            objects.append(item)
        return {
            "count": int(self.count),
            "method": self.method,
            "ok": self.ok,
            "error": self.error,
            "confidence": _num(self.confidence, 4),
            "by_label": dict(self.by_label),
            "image": {"width": int(width), "height": int(height), "kind": self.image_kind},
            "region": _json(self.region),
            "frame": self.frame,
            "crossings": {k: int(v) for k, v in self.crossings.items()},
            "objects": objects,
            "notes": list(self.notes),
            "details": _json({k: v for k, v in self.details.items() if k != "region_text"}),
        }
