"""Read a transcript from any of the shapes people actually have.

Accepted sources, in the order they are detected:

``list``/``tuple``
    of ``{"speaker", "text", "start"}`` dicts, or of :class:`Turn` objects.
path
    a ``str`` or ``pathlib.Path`` pointing at an existing ``.txt``, ``.vtt``,
    ``.srt`` or ``.md`` file.
WebVTT text
    a string beginning with ``WEBVTT``, or carrying ``-->`` cue arrows.
SubRip text
    numbered cue blocks with ``00:00:01,000 --> 00:00:04,000`` timings.
plain text
    ``Alice: ...`` lines, optionally stamped ``[00:01:02]``, or unlabelled prose.

The chosen shape is reported: :func:`parse_transcript` returns a plain ``list``
subclass whose ``source_format`` attribute names it, and the same string reaches
the caller as ``MeetingReport.source_format``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ._text import split_sentences, word_count

__all__ = ["Turn", "TurnList", "parse_transcript", "detect_format", "FORMATS"]

FORMATS = ("turns", "vtt", "srt", "labelled-text", "plain-text", "empty")

_TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".log"}
_CAPTION_SUFFIXES = {".vtt", ".srt", ".sbv"}
READABLE_SUFFIXES = _TEXT_SUFFIXES | _CAPTION_SUFFIXES

# 00:01:02.500 / 1:02.5 / 00:00:04,000
_TIME = r"\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?"
_CUE_ARROW = re.compile(r"(%s)\s*-+>\s*(%s)" % (_TIME, _TIME))
_TIME_ONLY = re.compile(r"^(%s)$" % _TIME)

# A leading stamp on a plain-text line: "[00:01:02] ", "(00:01:02) ", "00:01:02 - "
_LEADING_STAMP = re.compile(r"^[\[\(]?(%s)[\]\)]?\s*[-–—]?\s*" % _TIME)
# A stamp attached to the speaker: "Alice [00:01:02]:" / "Alice (00:01:02):"
_TRAILING_STAMP = re.compile(r"\s*[\[\(](%s)[\]\)]\s*$" % _TIME)

_VOICE_TAG = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+)>(.*?)(?:</v>|$)", re.IGNORECASE | re.DOTALL)
_ANY_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_SRT_INDEX = re.compile(r"^\d+$")

# Labels that look like "Name:" but are really section headers.
_NOT_A_SPEAKER = frozenset("""
note notes agenda summary decisions decision actions action attendees present
apologies date time location topic topics minutes recording transcript subject
steps todo follow followup background context purpose goal goals
warning warnings example examples output input result results
""".split())

_MAX_LABEL_WORDS = 5
_MAX_LABEL_CHARS = 48
# Unlabelled prose is cut into readable turns at these sizes.
_PARAGRAPH_SPLIT_WORDS = 80
_PARAGRAPH_GROUP_WORDS = 50


@dataclass
class Turn:
    """One contiguous stretch of speech by one person.

    ``start`` and ``end`` are seconds from the top of the meeting, or ``None``
    when the source carried no timings. ``speaker`` is ``None`` when the source
    carried no labels.
    """

    index: int
    speaker: Optional[str]
    text: str
    start: Optional[float] = None
    end: Optional[float] = None

    @property
    def words(self) -> int:
        """Number of word tokens spoken in this turn."""
        return word_count(self.text)

    @property
    def duration(self) -> Optional[float]:
        """Length of the turn in seconds, when the source was timed."""
        if self.start is None or self.end is None:
            return None
        return max(0.0, self.end - self.start)

    def sentences(self) -> List[str]:
        """The turn split into sentences."""
        return split_sentences(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the turn."""
        return {
            "index": self.index,
            "speaker": self.speaker,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": self.words,
        }


class TurnList(List[Turn]):
    """A plain ``list`` of :class:`Turn` that also remembers the source format."""

    source_format: str = "turns"

    def __init__(self, turns: Sequence[Turn] = (), source_format: str = "turns") -> None:
        super().__init__(turns)
        self.source_format = source_format

    @property
    def speakers(self) -> List[str]:
        """Distinct speaker names, in the order they first speak."""
        seen: List[str] = []
        for turn in self:
            if turn.speaker and turn.speaker not in seen:
                seen.append(turn.speaker)
        return seen

    @property
    def timed(self) -> bool:
        """True when at least one turn carries a start time."""
        return any(turn.start is not None for turn in self)


def parse_timestamp(raw: str) -> Optional[float]:
    """Seconds for ``HH:MM:SS.mmm``, ``MM:SS.mmm`` or ``HH:MM:SS,mmm``.

    Hours are optional and milliseconds may use either a dot or a comma, so
    WebVTT and SubRip stamps both land here.
    """
    if not raw:
        return None
    text = raw.strip().replace(",", ".")
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(parts) == 3:
        hours, minutes, seconds = numbers
    else:
        hours = 0.0
        minutes, seconds = numbers
    return hours * 3600.0 + minutes * 60.0 + seconds


def _looks_like_path(source: Any) -> bool:
    """True when ``source`` should be read from disk rather than parsed as text."""
    if isinstance(source, Path):
        return True
    if not isinstance(source, str):
        return False
    if "\n" in source or "\r" in source:
        return False
    if len(source) > 260:
        return False
    stripped = source.strip()
    if not stripped:
        return False
    suffix = os.path.splitext(stripped)[1].lower()
    return suffix in READABLE_SUFFIXES


def _read_path(source: Union[str, Path]) -> str:
    """Read a transcript file as UTF-8, tolerating a BOM and stray bytes."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError("transcript file not found: %r" % str(source))
    data = path.read_bytes()
    return data.decode("utf-8-sig", errors="replace")


def _is_speaker_label(label: str) -> bool:
    """Whether the text before a colon is a name rather than prose.

    Guards against lines like ``"We decided: Postgres"`` or ``"Action items:"``
    being read as a speaker, which is what makes unlabelled transcripts work.
    """
    label = label.strip()
    if not label or len(label) > _MAX_LABEL_CHARS:
        return False
    if label[-1] in ",;":
        return False
    for char in "?!\"()[]{}<>/|":
        if char in label:
            return False
    words = label.split()
    if not words or len(words) > _MAX_LABEL_WORDS:
        return False
    if label.lower() in _NOT_A_SPEAKER:
        return False
    if all(word.lower().strip(".,-_") in _NOT_A_SPEAKER for word in words):
        return False
    first = label[0]
    if not (first.isalpha() or first.isdigit() or first == "_" or first == "@"):
        return False
    for word in words:
        core = word.strip(".,-_@#'’")
        if not core:
            continue
        cased = [char for char in core if char.islower() or char.isupper()]
        if not cased:
            # Digits, or a script without case such as CJK: acceptable in a name.
            continue
        if not core[0].isupper():
            return False
    return True


def _split_labelled_line(line: str) -> Tuple[Optional[str], Optional[float], str]:
    """Pull ``(speaker, start, text)`` out of one plain-text line."""
    start = None  # type: Optional[float]
    match = _LEADING_STAMP.match(line)
    if match:
        start = parse_timestamp(match.group(1))
        line = line[match.end():]
    head, sep, tail = line.partition(":")
    if not sep:
        return None, start, line.strip()
    stamped = _TRAILING_STAMP.search(head)
    if stamped:
        if start is None:
            start = parse_timestamp(stamped.group(1))
        head = head[: stamped.start()]
    if _is_speaker_label(head):
        return head.strip(), start, tail.strip()
    return None, start, line.strip()


def _cue_blocks(text: str) -> List[List[str]]:
    """Split caption text into blocks on blank lines."""
    blocks = []  # type: List[List[str]]
    current = []  # type: List[str]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip():
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _clean_cue_text(lines: Sequence[str]) -> Tuple[Optional[str], str]:
    """Return ``(speaker, text)`` for the payload lines of one cue."""
    speaker = None  # type: Optional[str]
    parts = []  # type: List[str]
    for line in lines:
        voice = _VOICE_TAG.search(line)
        if voice:
            speaker = speaker or voice.group(1).strip() or None
            line = voice.group(2)
        line = _ANY_TAG.sub("", line)
        line = line.replace("&nbsp;", " ")
        line = line.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        line = line.strip()
        if not line:
            continue
        if speaker is None and not parts:
            label, _stamp, remainder = _split_labelled_line(line)
            if label:
                speaker = label
                line = remainder
                if not line:
                    continue
        parts.append(line)
    return speaker, " ".join(parts).strip()


def _parse_captions(text: str) -> List[Turn]:
    """Parse WebVTT or SubRip cues into turns."""
    turns = []  # type: List[Turn]
    for block in _cue_blocks(text):
        arrow_at = None
        for position, line in enumerate(block):
            if _CUE_ARROW.search(line):
                arrow_at = position
                break
        if arrow_at is None:
            continue
        match = _CUE_ARROW.search(block[arrow_at])
        if match is None:  # pragma: no cover - guarded by the loop above
            continue
        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        payload = block[arrow_at + 1:]
        speaker, body = _clean_cue_text(payload)
        if not body:
            continue
        turns.append(Turn(index=len(turns), speaker=speaker, text=body, start=start, end=end))
    if not turns:
        # Announced itself as a caption track but held no usable cue.
        return _parse_plain_text(_strip_caption_scaffolding(text))
    return turns


def _strip_caption_scaffolding(text: str) -> str:
    """Drop cue numbers, arrows and the WEBVTT header, leaving the words."""
    kept = []  # type: List[str]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if stripped.upper().startswith("WEBVTT"):
            continue
        if _CUE_ARROW.search(stripped) or _SRT_INDEX.match(stripped) or _TIME_ONLY.match(stripped):
            continue
        kept.append(_ANY_TAG.sub("", stripped))
    return "\n".join(kept)


def _group_long_paragraph(text: str) -> List[str]:
    """Cut an unlabelled wall of prose into turn-sized groups of sentences."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []
    groups = []  # type: List[str]
    current = []  # type: List[str]
    running = 0
    for sentence in sentences:
        current.append(sentence)
        running += word_count(sentence)
        if running >= _PARAGRAPH_GROUP_WORDS:
            groups.append(" ".join(current))
            current = []
            running = 0
    if current:
        if groups and running < 8:
            groups[-1] = groups[-1] + " " + " ".join(current)
        else:
            groups.append(" ".join(current))
    return groups


def _parse_plain_text(text: str) -> List[Turn]:
    """Parse ``Alice: ...`` lines, or unlabelled prose, into turns."""
    lines = text.splitlines()
    parsed = [_split_labelled_line(line) for line in lines]
    has_labels = any(label for label, _start, _body in parsed)
    turns = []  # type: List[Turn]
    if has_labels:
        # Text is accumulated in per-turn buffers rather than by repeated string
        # concatenation, so a file with thousands of continuation lines stays linear.
        buffers = []  # type: List[List[str]]
        for (label, start, body), raw in zip(parsed, lines):
            if not raw.strip():
                continue
            if label:
                turns.append(Turn(index=len(turns), speaker=label, text="", start=start))
                buffers.append([body] if body else [])
            elif turns:
                if body:
                    buffers[-1].append(body)
                if turns[-1].start is None and start is not None:
                    turns[-1].start = start
            elif body:
                turns.append(Turn(index=len(turns), speaker=None, text="", start=start))
                buffers.append([body])
        for turn, buffer in zip(turns, buffers):
            turn.text = " ".join(buffer).strip()
        return [turn for turn in turns if turn.text.strip()]

    # No labels anywhere: paragraphs become turns, long ones are grouped.
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        single = " ".join(paragraphs[0].split())
        if word_count(single) <= _PARAGRAPH_SPLIT_WORDS:
            return [Turn(index=0, speaker=None, text=single, start=None)]
        paragraphs = [single]
    for paragraph in paragraphs:
        flat = " ".join(paragraph.split())
        if word_count(flat) > _PARAGRAPH_SPLIT_WORDS:
            pieces = _group_long_paragraph(flat)
        else:
            pieces = [flat]
        for piece in pieces:
            if piece.strip():
                turns.append(Turn(index=len(turns), speaker=None, text=piece.strip(), start=None))
    return turns


def _as_seconds(value: Any) -> Optional[float]:
    """Accept seconds as a number, or a ``00:01:02.5`` string."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if ":" in stripped:
            return parse_timestamp(stripped)
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _turn_from_mapping(item: Dict[str, Any], index: int) -> Turn:
    """Build a :class:`Turn` from one ``{"speaker", "text", "start"}`` dict."""
    lowered = {}
    for key, value in item.items():
        lowered[str(key).lower()] = value
    text = lowered.get("text")
    if text is None:
        text = lowered.get("content")
    if text is None:
        text = lowered.get("utterance")
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    speaker = lowered.get("speaker")
    if speaker is None:
        speaker = lowered.get("name")
    if speaker is None:
        speaker = lowered.get("who")
    if speaker is not None and not isinstance(speaker, str):
        speaker = str(speaker)
    if isinstance(speaker, str):
        speaker = speaker.strip() or None
    start = lowered.get("start")
    if start is None:
        start = lowered.get("time")
    if start is None:
        start = lowered.get("timestamp")
    return Turn(
        index=index,
        speaker=speaker,
        text=" ".join(text.split()),
        start=_as_seconds(start),
        end=_as_seconds(lowered.get("end", lowered.get("stop"))),
    )


def _parse_sequence(source: Sequence[Any]) -> List[Turn]:
    """Parse a list of dicts, :class:`Turn` objects, or plain strings."""
    turns = []  # type: List[Turn]
    for item in source:
        if isinstance(item, Turn):
            turn = Turn(
                index=len(turns),
                speaker=item.speaker,
                text=item.text,
                start=item.start,
                end=item.end,
            )
        elif isinstance(item, dict):
            turn = _turn_from_mapping(item, len(turns))
        elif isinstance(item, str):
            label, start, body = _split_labelled_line(item)
            turn = Turn(index=len(turns), speaker=label, text=body, start=start)
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            turn = Turn(
                index=len(turns),
                speaker=str(item[0]) if item[0] is not None else None,
                text=str(item[1]),
                start=_as_seconds(item[2]) if len(item) > 2 else None,
            )
        else:
            raise TypeError(
                "each transcript turn must be a dict with 'speaker'/'text'/'start', "
                "a Turn, or a string; got %s" % type(item).__name__
            )
        if turn.text.strip():
            turn.index = len(turns)
            turns.append(turn)
    return turns


def _detect_text_format(text: str) -> str:
    """Classify transcript text by looking at its cue and label structure."""
    if not text or not text.strip():
        return "empty"
    stripped = text.lstrip("﻿ \t\r\n")
    if stripped[:6].upper() == "WEBVTT":
        return "vtt"
    arrows = _CUE_ARROW.findall(text)
    if arrows:
        # SubRip numbers each cue and uses a comma before the milliseconds.
        commas = 0
        for start, end in arrows:
            if "," in start or "," in end:
                commas += 1
        numbered = 0
        for block in _cue_blocks(text):
            if (
                len(block) >= 2
                and _SRT_INDEX.match(block[0].strip())
                and _CUE_ARROW.search(block[1])
            ):
                numbered += 1
        if commas or numbered >= max(1, len(arrows) // 2):
            return "srt"
        return "vtt"
    for line in text.splitlines():
        if line.strip() and _split_labelled_line(line)[0]:
            return "labelled-text"
    return "plain-text"


def detect_format(source: Any) -> str:
    """Name the transcript shape without keeping the parse.

    Returns one of :data:`FORMATS`. Files are classified by their content, so a
    ``.txt`` file holding WebVTT is still reported as ``"vtt"``.
    """
    if source is None:
        return "empty"
    if isinstance(source, TurnList):
        return source.source_format
    if isinstance(source, (list, tuple)):
        return "turns" if len(source) else "empty"
    if _looks_like_path(source):
        return _detect_text_format(_read_path(source))
    if isinstance(source, str):
        return _detect_text_format(source)
    raise TypeError(
        "transcript must be text, a list of turns, or a path to a .txt/.vtt/.srt "
        "file; got %s" % type(source).__name__
    )


def _merge_consecutive(turns: Sequence[Turn]) -> List[Turn]:
    """Join back-to-back cues from the same speaker into one real turn.

    Caption tracks break a single sentence across several cues; a meeting turn
    is the whole stretch, so merging is what makes monologue length and turn
    counts mean anything.
    """
    merged = []  # type: List[Turn]
    buffers = []  # type: List[List[str]]
    for turn in turns:
        if merged and merged[-1].speaker == turn.speaker:
            buffers[-1].append(turn.text)
            if turn.end is not None:
                merged[-1].end = turn.end
            continue
        merged.append(
            Turn(
                index=len(merged),
                speaker=turn.speaker,
                text="",
                start=turn.start,
                end=turn.end,
            )
        )
        buffers.append([turn.text])
    for turn, buffer in zip(merged, buffers):
        turn.text = " ".join(piece for piece in buffer if piece).strip()
    return merged


def parse_transcript(source: Any) -> TurnList:
    """Read ``source`` into a list of :class:`Turn` objects.

    ``source`` may be plain text, a list of ``{"speaker", "text", "start"}``
    dicts, a path to a ``.txt``/``.vtt``/``.srt`` file, or a WebVTT string. The
    shape that was used is reported on the returned list as ``source_format``.

    An empty or whitespace-only transcript returns an empty list rather than
    raising.
    """
    if source is None:
        return TurnList([], "empty")
    if isinstance(source, (list, tuple)):
        turns = _parse_sequence(source)
        return TurnList(turns, "turns" if turns else "empty")
    if _looks_like_path(source):
        text = _read_path(source)
    elif isinstance(source, str):
        text = source
    else:
        raise TypeError(
            "transcript must be text, a list of turns, or a path to a .txt/.vtt/.srt "
            "file; got %s" % type(source).__name__
        )

    source_format = _detect_text_format(text)
    if source_format == "empty":
        return TurnList([], "empty")
    if source_format in ("vtt", "srt"):
        turns = _merge_consecutive(_parse_captions(text))
    else:
        turns = _parse_plain_text(text)
    for position, turn in enumerate(turns):
        turn.index = position
    if not turns:
        return TurnList([], "empty")
    return TurnList(turns, source_format)
