"""Who spoke, how much, and who cut in.

Interruptions are counted two ways, because transcripts come both ways:

* timed sources -- a turn that starts before the previous speaker's cue ended
  is an interruption of that speaker;
* untimed sources -- a turn whose previous speaker was cut off mid-word, which
  transcription tools mark with a trailing dash or ellipsis.

Which rule was available is reported as ``MeetingReport.interruptions_basis``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._parse import Turn
from ._text import word_count

__all__ = ["Participation", "measure_participation"]

# A transcript marks a cut-off with a trailing dash or ellipsis.
_CUT_OFF = re.compile(r"(?:--|[-–—]|\.\.\.|…)\s*$")
# Timed overlap shorter than this is cue rounding, not a real interruption.
_OVERLAP_TOLERANCE = 0.25


@dataclass
class Participation:
    """One speaker's share of the meeting."""

    speaker: str
    turns: int = 0
    words: int = 0
    share: float = 0.0
    longest_monologue: int = 0
    longest_monologue_seconds: Optional[float] = None
    interruptions: int = 0
    interrupted_by_others: int = 0
    questions_asked: int = 0
    speaking_seconds: Optional[float] = None
    first_start: Optional[float] = None
    last_end: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of one speaker's participation."""
        return {
            "speaker": self.speaker,
            "turns": self.turns,
            "words": self.words,
            "share": round(self.share, 4),
            "longest_monologue": self.longest_monologue,
            "longest_monologue_seconds": self.longest_monologue_seconds,
            "interruptions": self.interruptions,
            "interrupted_by_others": self.interrupted_by_others,
            "questions_asked": self.questions_asked,
            "speaking_seconds": self.speaking_seconds,
            "first_start": self.first_start,
            "last_end": self.last_end,
        }

    def describe(self) -> str:
        """One plain-ASCII line summarising this speaker."""
        line = "%s: %d turn(s), %d words, %.0f%% of the talking" % (
            self.speaker,
            self.turns,
            self.words,
            self.share * 100.0,
        )
        if self.longest_monologue:
            line += ", longest run %d words" % self.longest_monologue
        if self.interruptions:
            line += ", interrupted %d time(s)" % self.interruptions
        return line


def measure_participation(
    turns: Sequence[Turn],
    questions: Sequence[Any] = (),
) -> Tuple[List[Participation], str]:
    """Per-speaker statistics; returns ``(rows, interruptions_basis)``.

    Rows come back heaviest speaker first. Turns with no speaker label are
    pooled under the name ``"(unlabelled)"`` so the word counts still add up to
    the whole meeting.
    """
    if not turns:
        return [], "none"

    unlabelled = "(unlabelled)"
    rows = {}  # type: Dict[str, Participation]
    order = []  # type: List[str]

    timed = any(turn.start is not None and turn.end is not None for turn in turns)
    basis = "timing overlap" if timed else "cut-off markers"
    if not timed and not any(_CUT_OFF.search(turn.text) for turn in turns):
        basis = "none available"

    for position, turn in enumerate(turns):
        name = turn.speaker or unlabelled
        if name not in rows:
            rows[name] = Participation(speaker=name)
            order.append(name)
        row = rows[name]
        row.turns += 1
        words = word_count(turn.text)
        row.words += words
        if words > row.longest_monologue:
            row.longest_monologue = words
            row.longest_monologue_seconds = turn.duration
        if turn.start is not None:
            if row.first_start is None or turn.start < row.first_start:
                row.first_start = turn.start
        if turn.end is not None:
            if row.last_end is None or turn.end > row.last_end:
                row.last_end = turn.end
        if turn.duration is not None:
            row.speaking_seconds = (row.speaking_seconds or 0.0) + turn.duration

        if position == 0:
            continue
        previous = turns[position - 1]
        previous_name = previous.speaker or unlabelled
        if previous_name == name:
            continue
        interrupted = False
        if timed and previous.end is not None and turn.start is not None:
            interrupted = turn.start < previous.end - _OVERLAP_TOLERANCE
        elif not timed:
            interrupted = bool(_CUT_OFF.search(previous.text))
        if interrupted:
            row.interruptions += 1
            rows[previous_name].interrupted_by_others += 1

    for question in questions:
        name = getattr(question, "speaker", None) or unlabelled
        if name in rows:
            rows[name].questions_asked += 1

    total_words = sum(row.words for row in rows.values())
    for row in rows.values():
        row.share = (row.words / total_words) if total_words else 0.0
        if row.speaking_seconds is not None:
            row.speaking_seconds = round(row.speaking_seconds, 3)

    ordered = sorted(
        (rows[name] for name in order),
        key=lambda row: (-row.words, -row.turns, row.speaker),
    )
    return ordered, basis
