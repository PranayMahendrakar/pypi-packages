"""Turn a meeting transcript into decisions, action items and a summary.

This package reads a **transcript**. It never opens, decodes or analyses audio:
getting text out of a recording is a different job, done before this package is
called. Feed it plain text, a list of turns, or a ``.txt``/``.vtt``/``.srt`` file.

    >>> import meeting_intelligence as mi
    >>> report = mi.analyse("Alice: We decided to go with Postgres.")
    >>> report.decisions[0].cue
    'we decided'

Nothing is downloaded and no model is loaded. Extraction is done with the cue
phrases published as :data:`DECISION_CUES`, :data:`ACTION_RULES` and
:data:`DUE_PATTERNS`, so every result can be traced back to the phrase that
produced it.
"""
from __future__ import annotations

from ._analyse import analyse, analyze
from ._extract import (
    ACTION_RULES,
    DECISION_CUES,
    DUE_PATTERNS,
    Action,
    Decision,
    Question,
    extract_actions,
    extract_decisions,
    extract_questions,
    find_due,
    unanswered,
)
from ._parse import FORMATS, Turn, TurnList, detect_format, parse_transcript
from ._participation import Participation, measure_participation
from ._report import MeetingReport
from ._summarise import format_duration
from ._topics import Topic, segment_topics

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # the one-liner
    "analyse",
    "analyze",
    "MeetingReport",
    # parsing
    "parse_transcript",
    "detect_format",
    "Turn",
    "TurnList",
    "FORMATS",
    # records
    "Action",
    "Decision",
    "Question",
    "Topic",
    "Participation",
    # the pieces, for people who want one of them on its own
    "extract_decisions",
    "extract_actions",
    "extract_questions",
    "segment_topics",
    "measure_participation",
    "find_due",
    "unanswered",
    "format_duration",
    # the published cue phrases
    "DECISION_CUES",
    "ACTION_RULES",
    "DUE_PATTERNS",
]
