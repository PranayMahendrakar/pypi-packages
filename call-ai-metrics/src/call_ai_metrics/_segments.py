"""Turn diarized segments, from any tool, into one clean speech timeline.

Accepted shapes, all of them read-only as far as the caller is concerned:

* ``[(start_s, end_s, speaker), ...]`` or ``[(start_s, end_s, speaker, text), ...]``
* a list of dicts with ``start``/``end``/``speaker`` (and optionally ``text``)
  keys, or of objects with those attributes
* a pandas DataFrame with those columns
* a ``.csv``/``.tsv`` file (with or without a header row) or a ``.json`` file
* anything with a ``segments`` attribute holding one of the above, and pyannote
  style annotations (anything with ``itertracks(yield_label=True)``)

Whatever arrives is validated, sorted into time order, and overlapping segments
from the same speaker are merged so no second is ever counted twice.
"""

from __future__ import annotations

import csv
import io
import json
import math
import numbers
import os
import re
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

EPS = 1e-9

_START_KEYS = ("start_s", "start", "begin", "start_time", "onset")
_END_KEYS = ("end_s", "end", "stop", "end_time", "offset")
_SPEAKER_KEYS = ("speaker", "label", "spk", "speaker_id", "party", "channel")
_TEXT_KEYS = ("text", "transcript", "words")

# Scripts written without spaces between words: each character counts as one word.
_NO_SPACE_SCRIPT = re.compile(
    "[぀-ヿ㐀-䶿一-鿿豈-﫿ㇰ-ㇿ]"
)
_WORDLIKE = re.compile(r"\w")


def count_words(text: str) -> int:
    """Count spoken words in a transcript snippet.

    Words are whitespace-separated tokens holding at least one letter or digit,
    so punctuation on its own is not a word. Chinese and Japanese are usually
    written without spaces; there each character counts as one word. When the
    transcript already separates them with spaces (two or more such tokens),
    each unbroken run counts as one word instead.
    """
    tokens = text.split()
    segmented = sum(1 for token in tokens if _NO_SPACE_SCRIPT.search(token)) >= 2
    total = 0
    for token in tokens:
        runs = _NO_SPACE_SCRIPT.findall(token)
        wide = len(runs)
        if segmented and runs:
            wide = len(re.findall(_NO_SPACE_SCRIPT.pattern + "+", token))
        rest = _NO_SPACE_SCRIPT.sub(" ", token)
        total += wide + sum(1 for part in rest.split() if _WORDLIKE.search(part))
    return total


class RawSegment:
    """One validated input segment, before merging."""

    __slots__ = ("start", "end", "label", "text", "index")

    def __init__(
        self, start: float, end: float, label: Hashable, text: Optional[str], index: int
    ) -> None:
        self.start = start
        self.end = end
        self.label = label
        self.text = text
        self.index = index


class Timeline:
    """Merged speech intervals per party, plus what was done to get them."""

    __slots__ = ("speakers", "intervals", "duration_s", "words", "transcribed_s", "notes", "given")

    def __init__(
        self,
        speakers: List[str],
        intervals: Dict[str, List[Tuple[float, float]]],
        duration_s: float,
        words: Dict[str, int],
        transcribed_s: Dict[str, float],
        notes: List[str],
        given: int,
    ) -> None:
        self.speakers = speakers
        self.intervals = intervals
        self.duration_s = duration_s
        self.words = words
        self.transcribed_s = transcribed_s
        self.notes = notes
        self.given = given


def merge_intervals(spans: Iterable[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], int]:
    """Union of intervals, and how many genuinely overlapped (not just touched)."""
    ordered = sorted(spans)
    merged: List[Tuple[float, float]] = []
    overlapped = 0
    for start, end in ordered:
        if merged and start <= merged[-1][1] + EPS:
            if start < merged[-1][1] - EPS:
                overlapped += 1
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged, overlapped


def _array_like(item: Any) -> bool:
    """A row of a numpy array or similar: indexable, sized, and not an object."""
    return (
        hasattr(item, "__len__")
        and hasattr(item, "__getitem__")
        and not any(hasattr(item, name) for name in _START_KEYS)
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if type(value).__name__ in ("NAType", "NaTType"):
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(value != value)  # NaN-like scalars (numpy, pandas NA)
    except (TypeError, ValueError):
        return False


def _as_time(value: Any, what: str, index: int) -> float:
    if isinstance(value, bool) or _is_missing(value):
        raise ValueError("segment {} has no usable {} time ({!r})".format(index, what, value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "segment {} has a {} time of {!r}, which is not a number of "
            "seconds".format(index, what, value)
        ) from None
    if not math.isfinite(number):
        raise ValueError("segment {} has a {} time of {}".format(index, what, number))
    return number


def _as_label(value: Any, index: int) -> Hashable:
    if _is_missing(value):
        raise ValueError("segment {} has no speaker".format(index))
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        return int(number) if number.is_integer() else number
    if isinstance(value, str):
        return value.strip()
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


def _as_text(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _lookup(mapping: Mapping[str, Any], keys: Sequence[str]) -> Tuple[bool, Any]:
    lowered: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, str):
            lowered.setdefault(key.strip().lower(), value)
    for key in keys:
        if key in lowered:
            return True, lowered[key]
    return False, None


def _from_mapping(item: Mapping[str, Any], index: int) -> Tuple[Any, Any, Any, Any]:
    found_start, start = _lookup(item, _START_KEYS)
    found_end, end = _lookup(item, _END_KEYS)
    found_speaker, speaker = _lookup(item, _SPEAKER_KEYS)
    _, text = _lookup(item, _TEXT_KEYS)
    if not (found_start and found_end and found_speaker):
        raise ValueError(
            "segment {} has keys {}; it needs start, end and speaker (text is "
            "optional)".format(index, sorted(str(key) for key in item.keys()))
        )
    return start, end, speaker, text


def _from_object(item: Any, index: int) -> Tuple[Any, Any, Any, Any]:
    def attribute(names: Sequence[str]) -> Tuple[bool, Any]:
        for name in names:
            if hasattr(item, name):
                return True, getattr(item, name)
        return False, None

    found_start, start = attribute(_START_KEYS)
    found_end, end = attribute(_END_KEYS)
    found_speaker, speaker = attribute(_SPEAKER_KEYS)
    _, text = attribute(_TEXT_KEYS)
    if not (found_start and found_end and found_speaker):
        raise ValueError(
            "segment {} is a {}; expected (start_s, end_s, speaker), "
            "(start_s, end_s, speaker, text), a dict with those keys, or an "
            "object with those attributes".format(index, type(item).__name__)
        )
    return start, end, speaker, text


def _coerce(item: Any, index: int) -> RawSegment:
    if isinstance(item, Mapping):
        start, end, speaker, text = _from_mapping(item, index)
    elif isinstance(item, (str, bytes)):
        raise ValueError(
            "segment {} is the string {!r}; expected (start_s, end_s, speaker)".format(
                index, item
            )
        )
    elif isinstance(item, Sequence) or _array_like(item):
        values = list(item)
        if len(values) not in (3, 4):
            raise ValueError(
                "segment {} has {} fields; expected (start_s, end_s, speaker) or "
                "(start_s, end_s, speaker, text)".format(index, len(values))
            )
        start, end, speaker = values[0], values[1], values[2]
        text = values[3] if len(values) == 4 else None
    else:
        start, end, speaker, text = _from_object(item, index)
    begin = _as_time(start, "start", index)
    finish = _as_time(end, "end", index)
    if begin < 0:
        raise ValueError(
            "segment {} starts at {} s; times are seconds from the start of the "
            "call and cannot be negative".format(index, begin)
        )
    if finish < begin:
        raise ValueError(
            "segment {} ends ({} s) before it starts ({} s)".format(index, finish, begin)
        )
    return RawSegment(begin, finish, _as_label(speaker, index), _as_text(text), index)


def _check_columns(columns: Sequence[Any], where: str) -> None:
    seen: Dict[str, int] = {}
    for column in columns:
        key = str(column)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(
            "{} has duplicate column names: {}; rename them so start, end, speaker "
            "and text are each one column".format(where, ", ".join(duplicates))
        )


def _column(columns: Sequence[Any], keys: Sequence[str]) -> Optional[Any]:
    by_name: Dict[str, Any] = {}
    for column in columns:
        by_name.setdefault(str(column).strip().lower(), column)
    for key in keys:
        if key in by_name:
            return by_name[key]
    return None


def rows_from_frame(frame: Any) -> List[Tuple[Any, Any, Any, Any]]:
    """Read segments out of a pandas DataFrame without writing to it."""
    columns = list(frame.columns)
    _check_columns(columns, "the DataFrame")
    start = _column(columns, _START_KEYS)
    end = _column(columns, _END_KEYS)
    speaker = _column(columns, _SPEAKER_KEYS)
    text = _column(columns, _TEXT_KEYS)
    if start is None or end is None or speaker is None:
        raise ValueError(
            "the DataFrame has columns {}; it needs start, end and speaker columns "
            "(text is optional)".format([str(column) for column in columns])
        )
    starts = frame[start].tolist()
    ends = frame[end].tolist()
    speakers = frame[speaker].tolist()
    texts = frame[text].tolist() if text is not None else [None] * len(starts)
    return list(zip(starts, ends, speakers, texts))


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def rows_from_csv(path: str) -> List[Tuple[Any, Any, Any, Any]]:
    """Read segments from a CSV or TSV file, with or without a header row."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        content = handle.read()
    if not content.strip():
        return []
    sample = content[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if path.lower().endswith(".tsv") else ","
    rows = [
        row
        for row in csv.reader(io.StringIO(content), delimiter=delimiter)
        if any(cell.strip() for cell in row)
    ]
    if not rows:
        return []
    first = rows[0]
    if len(first) >= 2 and _looks_numeric(first[0].strip()) and _looks_numeric(first[1].strip()):
        records = []
        for number, row in enumerate(rows):
            if len(row) < 3:
                raise ValueError(
                    "{}: row {} has {} fields; expected start, end, speaker and "
                    "optionally text".format(path, number + 1, len(row))
                )
            text = delimiter.join(row[3:]) if len(row) > 3 else None
            records.append((row[0], row[1], row[2], text))
        return records
    header = [cell.strip() for cell in first]
    _check_columns(header, os.path.basename(path))
    start = _column(header, _START_KEYS)
    end = _column(header, _END_KEYS)
    speaker = _column(header, _SPEAKER_KEYS)
    text = _column(header, _TEXT_KEYS)
    if start is None or end is None or speaker is None:
        raise ValueError(
            "{}: the header is {}; it needs start, end and speaker columns (text "
            "is optional)".format(path, header)
        )
    position = {name: index for index, name in enumerate(header)}
    records = []
    for row in rows[1:]:
        cells = row + [""] * (len(header) - len(row))
        records.append(
            (
                cells[position[start]].strip(),
                cells[position[end]].strip(),
                cells[position[speaker]].strip(),
                cells[position[text]] if text is not None else None,
            )
        )
    return records


def rows_from_json(path: str) -> List[Any]:
    """Read segments from a JSON list, or a JSON object with a ``segments`` list."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        if "segments" not in data:
            raise ValueError(
                "{}: the JSON object has no 'segments' list (keys: {})".format(
                    path, sorted(str(key) for key in data.keys())
                )
            )
        data = data["segments"]
    if not isinstance(data, list):
        raise ValueError("{}: expected a JSON list of segments".format(path))
    return data


def rows_from_annotation(annotation: Any) -> List[Tuple[Any, Any, Any, Any]]:
    """Read a pyannote-style annotation (``itertracks(yield_label=True)``)."""
    records = []
    for track in annotation.itertracks(yield_label=True):
        segment, label = track[0], track[-1]
        records.append((segment.start, segment.end, label, None))
    return records


def _placeholder(taken: Sequence[str], preferred: str) -> str:
    name = preferred
    suffix = 2
    while name in taken:
        name = "{}_{}".format(preferred, suffix)
        suffix += 1
    return name


def build_timeline(
    items: Sequence[Any],
    speakers: Any,
    *,
    notes: Optional[List[str]] = None,
) -> Timeline:
    """Validate, sort and merge segments into a :class:`Timeline`.

    Args:
        items: Segments in any accepted shape.
        speakers: ``None``, a list declaring the parties (and their order), or a
            mapping renaming labels, e.g. ``{0: "agent", 1: "customer"}``.
        notes: Notes already collected about the input, extended in place.
    """
    notes = list(notes or [])
    raw = [_coerce(item, index) for index, item in enumerate(items)]

    out_of_order = any(raw[i].start < raw[i - 1].start - EPS for i in range(1, len(raw)))
    if out_of_order:
        notes.append("segments arrived out of time order and were sorted")

    first_seen: List[Hashable] = []
    for segment in raw:
        if segment.label not in first_seen:
            first_seen.append(segment.label)

    rename: Dict[Hashable, str] = {}
    declared: List[str] = []
    if speakers is None:
        for label in first_seen:
            rename[label] = str(label)
    elif isinstance(speakers, Mapping):
        lookup: Dict[Any, str] = {}
        for key, value in speakers.items():
            lookup[key] = str(value)
            lookup[str(key)] = str(value)
        for label in first_seen:
            if label in lookup:
                rename[label] = lookup[label]
            elif str(label) in lookup:
                rename[label] = lookup[str(label)]
            else:
                rename[label] = str(label)
        for value in speakers.values():
            if str(value) not in declared:
                declared.append(str(value))
    elif isinstance(speakers, (str, bytes)):
        raise ValueError(
            "speakers={!r} is a single string; pass a list of names, e.g. "
            "['agent', 'customer'], or a dict renaming labels".format(speakers)
        )
    else:
        declared = [str(name) for name in speakers]
        if len(set(declared)) != len(declared):
            raise ValueError("speakers={} names someone twice".format(declared))
        known = set(declared)
        unknown = [label for label in first_seen if str(label) not in known]
        if unknown:
            raise ValueError(
                "the segments name {} but speakers={} does not; pass a dict to rename "
                "labels instead, e.g. {{{!r}: 'agent'}}".format(
                    ", ".join(repr(label) for label in unknown), declared, unknown[0]
                )
            )
        for label in first_seen:
            rename[label] = str(label)

    merged_labels: Dict[str, List[str]] = {}
    for label, name in rename.items():
        merged_labels.setdefault(name, []).append(str(label))
    for name, labels in merged_labels.items():
        if len(labels) > 1:
            notes.append(
                "labels {} were all named '{}' and are treated as one speaker".format(
                    ", ".join(labels), name
                )
            )

    order: List[str] = []
    for name in declared:
        if name not in order:
            order.append(name)
    for label in first_seen:
        if rename[label] not in order:
            order.append(rename[label])

    spans: Dict[str, List[Tuple[float, float]]] = {name: [] for name in order}
    texted: Dict[str, List[Tuple[float, float]]] = {name: [] for name in order}
    words: Dict[str, int] = {}
    zero_length = 0
    duration = 0.0
    for segment in sorted(raw, key=lambda seg: (seg.start, seg.end, seg.index)):
        name = rename[segment.label]
        duration = max(duration, segment.end)
        if segment.end - segment.start <= EPS:
            zero_length += 1
            continue
        spans[name].append((segment.start, segment.end))
        if segment.text is not None:
            texted[name].append((segment.start, segment.end))
            words[name] = words.get(name, 0) + count_words(segment.text)
    if zero_length:
        notes.append(
            "{} zero-length segment{} carried no speech time and {} ignored".format(
                zero_length, "" if zero_length == 1 else "s", "was" if zero_length == 1 else "were"
            )
        )

    intervals: Dict[str, List[Tuple[float, float]]] = {}
    overlapped_total = 0
    for name in order:
        merged, overlapped = merge_intervals(spans[name])
        intervals[name] = merged
        overlapped_total += overlapped
    if overlapped_total:
        notes.append(
            "{} overlapping segment{} from the same speaker {} merged, so no second "
            "is counted twice".format(
                overlapped_total,
                "" if overlapped_total == 1 else "s",
                "was" if overlapped_total == 1 else "were",
            )
        )

    transcribed: Dict[str, float] = {}
    for name in words:
        merged, _ = merge_intervals(texted[name])
        transcribed[name] = sum(end - start for start, end in merged)

    if len(order) < 2:
        defaults = ["A", "B"] if not order else ["other"]
        while len(order) < 2:
            name = _placeholder(order, defaults[0] if len(order) == 0 else defaults[-1])
            order.append(name)
            intervals[name] = []
        if raw:
            notes.append(
                "only one speaker appears in the segments, so '{}' was added with no "
                "talk time to keep this a two-party report".format(order[-1])
            )

    return Timeline(order, intervals, duration, words, transcribed, notes, len(raw))
