"""The one function most callers need: :func:`analyse`.

It parses whatever the caller has, runs every extractor over the turns, and
assembles a :class:`~meeting_intelligence._report.MeetingReport`. No model is
loaded, nothing is downloaded, and no audio is touched at any point -- the input
is always text that somebody else already transcribed.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence

from ._extract import extract_actions, extract_decisions, extract_questions
from ._parse import Turn, TurnList, parse_transcript
from ._participation import measure_participation
from ._report import MeetingReport
from ._summarise import build_summary_text, polish_with_llm
from ._text import word_count
from ._topics import segment_topics

logger = logging.getLogger(__name__)

__all__ = ["analyse", "analyze"]


def _clean_speakers(speakers: Any) -> List[str]:
    """Validate the caller's speaker list into a clean list of names."""
    if speakers is None:
        return []
    if isinstance(speakers, str):
        raise ValueError(
            "speakers must be a list of names, not a single string; "
            "pass ['Alice', 'Bob'] rather than 'Alice, Bob'"
        )
    try:
        items = list(speakers)
    except TypeError:
        raise ValueError(
            "speakers must be a list of names; got %s" % type(speakers).__name__
        ) from None
    names = []  # type: List[str]
    for item in items:
        if item is None:
            continue
        if not isinstance(item, str):
            item = str(item)
        item = item.strip()
        if item and item not in names:
            names.append(item)
    return names


def _duration(turns: Sequence[Turn]) -> Optional[float]:
    """Seconds from the first timestamp to the last, or ``None`` when untimed."""
    starts = [turn.start for turn in turns if turn.start is not None]
    if not starts:
        return None
    ends = [turn.end for turn in turns if turn.end is not None]
    last = max(ends) if ends else max(starts)
    span = float(last) - float(min(starts))
    return round(span, 3) if span > 0 else 0.0


def analyse(
    transcript: Any,
    *,
    speakers: Optional[Sequence[str]] = None,
    llm: Optional[Callable[[str], str]] = None,
) -> MeetingReport:
    """Turn a meeting transcript into decisions, action items and a summary.

    Parameters
    ----------
    transcript:
        Plain text, a list of ``{"speaker", "text", "start"}`` turns, a path to a
        ``.txt``/``.vtt``/``.srt`` file, or a WebVTT/SubRip string. The shape
        that was used is reported as ``report.source_format``.
    speakers:
        Optional roster of names. Owners found in the text are resolved against
        it, which fixes first-name-only mentions ("Sam will ..." for
        "Sam Okafor"). Names that only appear in the roster are still listed on
        the report. Defaults to the speakers found in the transcript.
    llm:
        Optional ``callable(prompt) -> str`` used only to improve the summary
        wording and the phrasing of action items. It is never required: without
        it the report is built entirely from documented cue phrases. If the
        callable raises or returns something unusable, the extractive result is
        kept and a note is added to ``report.warnings``.

    Returns
    -------
    MeetingReport
        Call ``report.summary()`` for text, ``report.to_dict()`` for JSON or
        ``report.to_markdown()`` for minutes. An empty transcript returns an
        empty report rather than raising.

    Examples
    --------
    >>> report = analyse("Alice: We decided to go with Postgres.")
    >>> report.decisions[0].text
    'We decided to go with Postgres.'
    """
    roster = _clean_speakers(speakers)
    turns = parse_transcript(transcript)
    if not isinstance(turns, TurnList):  # pragma: no cover - parse_transcript always returns one
        turns = TurnList(turns, "turns")

    warnings = []  # type: List[str]
    known = list(roster)
    for name in turns.speakers:
        if name not in known:
            known.append(name)

    if not turns:
        report = MeetingReport(
            summary_text=build_summary_text([]),
            source_format=turns.source_format,
            speakers=known,
        )
        report.warnings.append(
            "the transcript was empty or held no readable speech; an empty report was returned"
        )
        return report

    if roster and not turns.speakers:
        warnings.append(
            "speakers were supplied but the transcript carries no speaker labels; "
            "action owners are left unassigned"
        )

    decisions = extract_decisions(turns)
    actions = extract_actions(turns, known)
    questions = extract_questions(turns)
    topics, topic_warnings = segment_topics(turns)
    warnings.extend(topic_warnings)
    participation, basis = measure_participation(turns, questions)
    duration = _duration(turns)

    if not turns.speakers:
        warnings.append(
            "no speaker labels in this transcript; participation is pooled and "
            "action owners are left as None"
        )
    if basis == "none available":
        warnings.append(
            "no timings and no cut-off markers, so interruptions could not be counted"
        )

    summary_text = build_summary_text(
        turns, topics, decisions, actions, questions, participation, duration
    )

    llm_used = False
    if llm is not None:
        new_summary, rewritten, llm_warnings = polish_with_llm(
            llm, summary_text, decisions, actions
        )
        warnings.extend(llm_warnings)
        if new_summary:
            summary_text = new_summary
            llm_used = True
        for index, text in rewritten.items():
            if 0 <= index < len(actions):
                actions[index].source_text = actions[index].text
                actions[index].text = text
                llm_used = True

    return MeetingReport(
        summary_text=summary_text,
        decisions=decisions,
        action_items=actions,
        questions=questions,
        topics=topics,
        participation=participation,
        duration=duration,
        turns=turns,
        speakers=known,
        source_format=turns.source_format,
        n_turns=len(turns),
        n_words=sum(word_count(turn.text) for turn in turns),
        interruptions_basis=basis,
        llm_used=llm_used,
        warnings=warnings,
    )


#: US spelling of :func:`analyse`; the same function under a second name.
analyze = analyse
