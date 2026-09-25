"""Shift schedules: which hours of the day the line is supposed to run, split into shifts.

``shift_hours`` is accepted in whichever form is closest to hand::

    (6, 22)                                        one scheduled window a day, 06:00-22:00
    [(6, 14), (14, 22)]                            two shifts; other hours are unscheduled
    {"A": (6, 14), "B": (14, 22), "C": (22, 6)}    named shifts; C runs past midnight
    "06:00-14:00,14:00-22:00"                      the same as text (what the CLI takes)
    "A=6-14,B=14-22"                               named, as text
    8                                              8-hour shifts round the clock from 06:00

Times are hours (``6``, ``6.5``) or ``"HH:MM"`` text, in the timestamps' own local time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

MINUTES_PER_DAY = 24 * 60
DEFAULT_FIRST_SHIFT_START = 6 * 60  # most plants start the first shift at 06:00


@dataclass(frozen=True)
class Shift:
    """One shift: a name and a window of the day in minutes after midnight.

    ``end`` at or before ``start`` means the shift runs past midnight; ``end == start``
    is a 24-hour shift.
    """

    name: str
    start: int
    end: int

    @property
    def minutes(self) -> int:
        """Length of the shift in minutes."""
        length = (self.end - self.start) % MINUTES_PER_DAY
        return length or MINUTES_PER_DAY

    @property
    def window(self) -> str:
        """The shift's hours as ``"06:00-14:00"``."""
        return f"{clock(self.start)}-{clock(self.end)}"


@dataclass(frozen=True)
class Schedule:
    """The shifts of a day, and whether the caller gave them (``given``) or they are defaults."""

    shifts: Tuple[Shift, ...]
    given: bool

    @property
    def scheduled_minutes_per_day(self) -> int:
        """How many minutes of each day fall inside a shift."""
        return int(sum(s.minutes for s in self.shifts))

    @property
    def covers_whole_day(self) -> bool:
        """True when every minute of the day belongs to some shift."""
        return self.scheduled_minutes_per_day >= MINUTES_PER_DAY

    def lookup(self) -> np.ndarray:
        """Array of 1440 entries: the shift index for each minute of the day, -1 if none."""
        table = np.full(MINUTES_PER_DAY, -1, dtype=np.int64)
        for index, shift in enumerate(self.shifts):
            minutes = (shift.start + np.arange(shift.minutes)) % MINUTES_PER_DAY
            table[minutes] = index
        return table

    def describe(self) -> str:
        """One line of plain text, e.g. ``"2 shifts: 06:00-14:00, 14:00-22:00"``."""
        parts = []
        for shift in self.shifts:
            parts.append(shift.window if shift.name == shift.window else f"{shift.name} {shift.window}")
        noun = "shift" if len(self.shifts) == 1 else "shifts"
        text = f"{len(self.shifts)} {noun}: {', '.join(parts)}"
        if not self.given:
            text += " (default 8-hour blocks; every hour counts as scheduled)"
        elif self.covers_whole_day:
            text += " (round the clock)"
        return text


def clock(minute: int) -> str:
    """Minutes after midnight as ``"HH:MM"``."""
    minute = int(minute) % MINUTES_PER_DAY
    return f"{minute // 60:02d}:{minute % 60:02d}"


_TIME_TEXT = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(?:h|hrs?)?\s*$", re.I)
_HINT = "use hours like 6 or 22.5, or text like '06:00'"


def _minute_of_day(value: Any, spec: Any) -> int:
    """6 -> 360, 6.5 -> 390, "06:30" -> 390, 24 -> 1440 (the end of the day)."""
    if isinstance(value, bool):
        raise ValueError(f"shift_hours {spec!r}: {value!r} is not a time of day; {_HINT}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        hours = float(value)
    elif isinstance(value, str):
        match = _TIME_TEXT.match(value)
        if match:
            if match.group(2) is not None and int(match.group(2)) >= 60:
                raise ValueError(f"shift_hours {spec!r}: {value!r} has more than 59 minutes")
            hours = int(match.group(1)) + int(match.group(2) or 0) / 60.0
        else:
            try:
                hours = float(value)
            except ValueError:
                raise ValueError(
                    f"shift_hours {spec!r}: {value!r} is not a time of day; {_HINT}"
                ) from None
    else:
        raise ValueError(f"shift_hours {spec!r}: {value!r} is not a time of day; {_HINT}")
    if not np.isfinite(hours) or hours < 0 or hours > 24:
        raise ValueError(f"shift_hours {spec!r}: {value!r} is outside 0-24 hours")
    return int(round(hours * 60))


def _is_time_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _looks_like_window(value: Any) -> bool:
    return isinstance(value, str) and bool(re.search(r"\d\s*(?:-|to|\N{EN DASH})\s*\d", value))


def _pair(item: Any, spec: Any) -> Tuple[int, int]:
    if isinstance(item, str):
        text = item.strip().replace("\N{EN DASH}", "-").replace("\N{EM DASH}", "-")
        pieces = [p for p in re.split(r"\s*(?:-|\bto\b)\s*", text) if p != ""]
        if len(pieces) != 2:
            raise ValueError(
                f"shift_hours {spec!r}: {item!r} should look like '06:00-14:00' or '6-14'"
            )
        item = pieces
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        raise ValueError(
            f"shift_hours {spec!r}: each shift needs a (start, end) pair, got {item!r}"
        )
    return _minute_of_day(item[0], spec), _minute_of_day(item[1], spec)


def _blocks(hours: float, spec: Any) -> List[Tuple[str, int, int]]:
    length = int(round(float(hours) * 60)) if np.isfinite(float(hours)) else 0
    if length <= 0 or length > MINUTES_PER_DAY or MINUTES_PER_DAY % length:
        raise ValueError(
            f"shift_hours={spec!r}: a single number is the length of a shift in hours and "
            "must divide the day evenly (1, 2, 3, 4, 6, 8, 12 or 24)"
        )
    out = []
    for k in range(MINUTES_PER_DAY // length):
        start = (DEFAULT_FIRST_SHIFT_START + k * length) % MINUTES_PER_DAY
        out.append(("", start, (start + length) % MINUTES_PER_DAY))
    return out


def _parse_text(spec: str) -> List[Tuple[str, str]]:
    """'A=6-14, B=14-22' -> [('A', '6-14'), ('B', '14-22')]."""
    items: List[Tuple[str, str]] = []
    for part in re.split(r"[,;]", spec):
        part = part.strip()
        if not part:
            continue
        name, sep, window = part.partition("=")
        items.append((name.strip(), window.strip()) if sep else ("", part))
    if not items:
        raise ValueError(f"shift_hours {spec!r} names no shifts")
    return items


def parse_shift_hours(spec: Any) -> Schedule:
    """Turn any accepted ``shift_hours`` form into a :class:`Schedule`.

    ``None`` gives three default 8-hour blocks from 06:00 that are all scheduled, so every
    hour of the day counts; the report says so. Raises ``ValueError`` for overlapping
    shifts or a value that is not a time of day.
    """
    if isinstance(spec, Schedule):
        return spec
    if spec is None:
        blocks = _blocks(8, 8)
        return Schedule(
            shifts=tuple(Shift(f"{clock(s)}-{clock(e)}", s, e) for _, s, e in blocks),
            given=False,
        )

    named: List[Tuple[str, int, int]] = []
    if isinstance(spec, (int, float, np.integer, np.floating)) and not isinstance(spec, bool):
        named = _blocks(float(spec), spec)
    elif isinstance(spec, str):
        if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", spec):
            named = _blocks(float(spec), spec)
        else:
            for name, window in _parse_text(spec):
                start, end = _pair(window, spec)
                named.append((name, start, end))
    elif isinstance(spec, dict):
        if not spec:
            raise ValueError("shift_hours is an empty dict; give at least one shift")
        for name, window in spec.items():
            start, end = _pair(window, spec)
            named.append((str(name), start, end))
    elif isinstance(spec, (tuple, list)):
        items: Sequence[Any] = spec
        if len(items) == 0:
            raise ValueError("shift_hours is empty; give at least one (start, end) pair")
        if len(items) == 2 and all(_is_time_scalar(v) and not _looks_like_window(v) for v in items):
            items = [tuple(items)]
        for item in items:
            start, end = _pair(item, spec)
            named.append(("", start, end))
    else:
        raise ValueError(
            "shift_hours must be a (start, end) pair, a list or dict of pairs, text like "
            "'06:00-14:00,14:00-22:00', or a shift length in hours; "
            f"got {type(spec).__name__}"
        )

    shifts = []
    for name, start, end in named:
        start %= MINUTES_PER_DAY
        end %= MINUTES_PER_DAY
        if start == end and len(named) > 1:
            raise ValueError(
                f"shift_hours {spec!r}: a shift from {clock(start)} to {clock(end)} has no "
                "length; a 24-hour shift must be the only one"
            )
        shifts.append(Shift(name or f"{clock(start)}-{clock(end)}", start, end))
    _check_overlap(shifts, spec)
    names = [s.name for s in shifts]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"shift_hours {spec!r}: duplicate shift names {dupes}")
    shifts.sort(key=lambda s: ((s.start - DEFAULT_FIRST_SHIFT_START) % MINUTES_PER_DAY, s.name))
    return Schedule(shifts=tuple(shifts), given=True)


def _check_overlap(shifts: Iterable[Shift], spec: Any) -> None:
    owner: List[Optional[str]] = [None] * MINUTES_PER_DAY
    for shift in shifts:
        for step in range(shift.minutes):
            minute = (shift.start + step) % MINUTES_PER_DAY
            if owner[minute] is not None:
                raise ValueError(
                    f"shift_hours {spec!r}: shifts {owner[minute]} and {shift.window} overlap "
                    f"at {clock(minute)}"
                )
            owner[minute] = shift.window
