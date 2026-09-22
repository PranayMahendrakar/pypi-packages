"""Read a list of texts from a list, or from a .txt / .csv / .jsonl file."""
from __future__ import annotations

import csv

#: Largest single CSV field we will read, in characters.
_MAX_CSV_FIELD = 50_000_000
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["load_texts", "write_texts", "looks_like_path", "TEXT_SUFFIXES"]

TEXT_SUFFIXES = (".txt", ".csv", ".tsv", ".jsonl", ".ndjson", ".json")


def looks_like_path(obj: Any) -> bool:
    """True when ``obj`` should be read as a file rather than used as texts."""
    return isinstance(obj, (str, os.PathLike))


def _pick_column(header: Sequence[str], rows: Sequence[Sequence[str]]) -> int:
    """Choose the text column: ``text``-ish name first, else the wordiest one."""
    lowered = [str(name).strip().casefold() for name in header]
    for preferred in ("text", "content", "body", "message", "sentence", "passage", "description"):
        if preferred in lowered:
            return lowered.index(preferred)
    if len(header) == 1:
        return 0
    widths = []
    for column in range(len(header)):
        values = [row[column] for row in rows if column < len(row)]
        widths.append(sum(len(value) for value in values) / max(len(values), 1))
    return int(max(range(len(header)), key=lambda c: (widths[c], -c)))


def _load_csv(path: Path, column: Optional[str]) -> List[str]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        try:
            rows = list(csv.reader(handle, delimiter=delimiter))
        except csv.Error as exc:
            # Python caps a single CSV field at 131,072 characters by default, and a
            # long passage in one cell trips it. Raising the cap once is the whole fix;
            # a raw _csv.Error traceback told the user nothing about which file it was.
            csv.field_size_limit(_MAX_CSV_FIELD)
            handle.seek(0)
            try:
                rows = list(csv.reader(handle, delimiter=delimiter))
            except csv.Error as exc2:
                raise ValueError(
                    "could not read {0!r} as CSV: {1}. A single field larger than "
                    "{2:,} characters cannot be read.".format(path, exc2, _MAX_CSV_FIELD)
                ) from exc
    rows = [row for row in rows if row]
    if not rows:
        return []
    header = rows[0]
    body = rows[1:]
    if column is not None:
        lowered = [str(name).strip().casefold() for name in header]
        wanted = column.strip().casefold()
        if wanted not in lowered:
            raise ValueError(
                f"column {column!r} is not in {path.name}; available: {', '.join(header)}"
            )
        index = lowered.index(wanted)
    else:
        index = _pick_column(header, body)
        logger.info("reading column %r from %s", header[index], path.name)
    if not body:
        return []
    return [row[index] if index < len(row) else "" for row in body]


def _text_from_json_object(obj: Any, column: Optional[str], line_no: int, name: str) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if column is not None:
            if column not in obj:
                raise ValueError(f"{name} line {line_no}: no key {column!r}")
            return str(obj[column])
        for preferred in ("text", "content", "body", "message", "sentence", "passage"):
            if preferred in obj:
                return str(obj[preferred])
        for value in obj.values():
            if isinstance(value, str):
                return value
        raise ValueError(f"{name} line {line_no}: no string value to use as text")
    return str(obj)


def _load_jsonl(path: Path, column: Optional[str]) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    stripped = content.lstrip()
    if stripped.startswith("["):
        # A single JSON array is a friendly thing to accept here too.
        for position, item in enumerate(json.loads(stripped), start=1):
            out.append(_text_from_json_object(item, column, position, path.name))
        return out
    for line_no, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_no} is not valid JSON: {exc}") from exc
        out.append(_text_from_json_object(obj, column, line_no, path.name))
    return out


def load_texts(source: Any, column: Optional[str] = None) -> List[str]:
    """Return a list of strings from a list/iterable of strings or a file path.

    Supported files: ``.txt`` (one text per line), ``.csv``/``.tsv`` (a header
    row plus one text column) and ``.jsonl``/``.json`` (one object per line, or
    one JSON array). ``column`` names the csv column or json key to use.
    """
    if looks_like_path(source):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{str(path)!r} does not exist")
        if path.is_dir():
            raise ValueError(
                f"{str(path)!r} is a directory, not a .txt/.csv/.jsonl file"
            )
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            return _load_csv(path, column)
        if suffix in (".jsonl", ".ndjson", ".json"):
            return _load_jsonl(path, column)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read().splitlines()

    if isinstance(source, (bytes, bytearray)):
        raise TypeError("texts must be a list of str or a path, not bytes")
    if isinstance(source, Mapping):
        raise TypeError(
            "texts must be a list of strings or a path to a .txt/.csv/.jsonl "
            "file, not a dict (iterating one yields its keys, not its texts); "
            "pass list(mapping.values()) if the values are the texts"
        )
    if isinstance(source, (set, frozenset)):
        raise TypeError(
            "texts must be a list of strings or a path to a .txt/.csv/.jsonl "
            "file, not a set (a set has no stable order, so the reported "
            "indices would differ between runs); pass sorted(texts) instead"
        )
    if not isinstance(source, Sequence) or isinstance(source, (str,)):
        try:
            source = list(source)
        except TypeError as exc:
            raise TypeError(
                "texts must be a list of strings or a path to a .txt/.csv/.jsonl file"
            ) from exc

    out: List[str] = []
    for position, item in enumerate(source):
        if not isinstance(item, str):
            raise ValueError(
                f"texts[{position}] is {type(item).__name__}, expected str "
                "(every item must already be text)"
            )
        out.append(item)
    return out


def write_texts(path: str, texts: Iterable[str]) -> None:
    """Write one text per line, always UTF-8."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            handle.write(text.replace("\n", " ") + "\n")
