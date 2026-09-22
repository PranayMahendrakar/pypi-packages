"""Write the summary paragraph, and optionally let a caller's LLM polish it.

The package is fully useful with no LLM. The summary here is extractive: it is
built out of the meeting's own sentences, scored by how much of the meeting's
vocabulary each one carries, plus a plain count of what was decided and left
open. When a callable is supplied it is asked to rewrite that paragraph and the
action lines, and everything it does -- including raising, returning nothing, or
returning something unusable -- is contained in this module.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ._extract import Action, Decision, Question
from ._parse import Turn
from ._text import content_words, fold, truncate, word_count

logger = logging.getLogger(__name__)

__all__ = ["build_summary_text", "format_duration", "join_names", "key_sentences", "polish_with_llm"]

_MIN_SENTENCE_WORDS = 6
_MAX_SENTENCE_WORDS = 45
_KEY_SENTENCES = 3
_MAX_REDUNDANCY = 0.5

_MAX_LLM_ACTIONS = 25
_MAX_LLM_SUMMARY_CHARS = 4000
_MAX_LLM_ACTION_CHARS = 240
_NUMBERED_LINE = re.compile(r"^\s*(\d{1,3})\s*[.)\]:-]\s*(.+?)\s*$")

UNLABELLED = "(unlabelled)"


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Human wording for a span in seconds, or ``None`` when untimed."""
    if seconds is None:
        return None
    total = int(round(float(seconds)))
    if total < 0:
        return None
    if total < 90:
        return "%d second%s" % (total, "" if total == 1 else "s")
    minutes = total // 60
    if minutes < 60:
        return "%d minutes" % minutes
    hours, rest = divmod(minutes, 60)
    if rest:
        return "%d h %d min" % (hours, rest)
    return "%d hour%s" % (hours, "" if hours == 1 else "s")


def join_names(names: Sequence[str], limit: int = 4) -> str:
    """Join names for prose: "Alice, Bob and Carla", shortened when long."""
    clean = [str(name) for name in names if name]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) <= limit:
        return "%s and %s" % (", ".join(clean[:-1]), clean[-1])
    extra = len(clean) - limit
    return "%s and %d other%s" % (", ".join(clean[:limit]), extra, "" if extra == 1 else "s")


def _plural(count: int, noun: str) -> str:
    """Count with its noun, pluralised: 1 topic, 3 topics."""
    return "%d %s%s" % (count, noun, "" if count == 1 else "s")


# --------------------------------------------------------------------------
# Extractive key sentences
# --------------------------------------------------------------------------


def key_sentences(
    turns: Sequence[Turn],
    limit: int = _KEY_SENTENCES,
    exclude: Sequence[str] = (),
) -> List[str]:
    """The sentences that carry the most of the meeting's own vocabulary.

    Scored by summed log-frequency of their content words, normalised so long
    sentences do not win by length alone, then picked greedily while skipping
    anything that repeats a sentence already chosen. Returned in transcript
    order, not score order, so they still read as a narrative.
    """
    if limit <= 0 or not turns:
        return []

    skip = set()
    for item in exclude:
        if item:
            skip.add(fold(item))

    frequency = Counter()  # type: Counter
    candidates = []  # type: List[Tuple[int, str, List[str]]]
    position = 0
    for turn in turns:
        for sentence in turn.sentences():
            stripped = sentence.strip()
            position += 1
            words = content_words(stripped)
            frequency.update(words)
            if not words or len(set(words)) < 4:
                continue
            length = word_count(stripped)
            if length < _MIN_SENTENCE_WORDS or length > _MAX_SENTENCE_WORDS:
                continue
            if stripped.endswith("?"):
                continue
            if fold(stripped) in skip:
                continue
            candidates.append((position, stripped, words))

    if not candidates:
        return []

    scored = []  # type: List[Tuple[float, int, str, set]]
    for order, sentence, words in candidates:
        unique = set(words)
        weight = sum(1.0 + math.log(frequency[word]) for word in unique)
        score = weight / (len(unique) ** 0.6)
        scored.append((score, order, sentence, unique))
    scored.sort(key=lambda item: (-item[0], item[1]))

    picked = []  # type: List[Tuple[int, str]]
    taken = []  # type: List[set]
    for _score, order, sentence, unique in scored:
        redundant = False
        for previous in taken:
            union = unique | previous
            if union and len(unique & previous) / len(union) > _MAX_REDUNDANCY:
                redundant = True
                break
        if redundant:
            continue
        picked.append((order, sentence))
        taken.append(unique)
        if len(picked) >= limit:
            break
    picked.sort(key=lambda item: item[0])
    return [sentence for _order, sentence in picked]


# --------------------------------------------------------------------------
# The summary paragraph
# --------------------------------------------------------------------------


def build_summary_text(
    turns: Sequence[Turn],
    topics: Sequence[Any] = (),
    decisions: Sequence[Decision] = (),
    actions: Sequence[Action] = (),
    questions: Sequence[Question] = (),
    participation: Sequence[Any] = (),
    duration: Optional[float] = None,
) -> str:
    """A short paragraph describing the meeting, built without any model."""
    if not turns:
        return "Empty transcript: no turns were found, so there is nothing to report."

    sentences = []  # type: List[str]

    named = [row.speaker for row in participation if row.speaker and row.speaker != UNLABELLED]
    turn_count = _plural(len(turns), "turn")
    span = format_duration(duration)
    if named:
        opening = "%s spoke across %s" % (join_names(named), turn_count)
    else:
        opening = "An unlabelled transcript of %s" % turn_count
    if span:
        opening += " (about %s)" % span
    sentences.append(opening + ".")

    if topics:
        labels = "; ".join(topic.label for topic in topics[:4])
        if len(topics) > 4:
            labels += "; ..."
        sentences.append("%s: %s." % (_plural(len(topics), "topic"), labels))

    ranked = sorted(decisions, key=lambda item: -item.confidence)[:2]
    if ranked:
        sentences.append("Decided: " + " ".join(truncate(item.text, 140) for item in ranked))

    highlights = key_sentences(turns, _KEY_SENTENCES, exclude=[item.text for item in decisions])
    if highlights:
        sentences.append(" ".join(truncate(text, 180) for text in highlights))

    closing = []  # type: List[str]
    if actions:
        owned = sum(1 for action in actions if action.owner)
        dated = sum(1 for action in actions if action.due)
        detail = []  # type: List[str]
        if owned:
            detail.append("%d with an owner" % owned)
        if dated:
            detail.append("%d with a due date" % dated)
        line = _plural(len(actions), "action item")
        if detail:
            line += " (%s)" % ", ".join(detail)
        closing.append(line)
    else:
        closing.append("no action items")
    still_open = [item for item in questions if not item.answered]
    if still_open:
        closing.append("%s left open" % _plural(len(still_open), "question"))
    elif questions:
        closing.append("every question asked got an answer")
    if not decisions:
        closing.append("no decision was recorded")
    sentences.append(_capitalise(", ".join(closing)) + ".")

    return " ".join(part for part in sentences if part.strip())


def _capitalise(text: str) -> str:
    """Upper-case the first letter, leaving the rest alone."""
    if not text:
        return text
    return text[0].upper() + text[1:]


# --------------------------------------------------------------------------
# Optional LLM polish
# --------------------------------------------------------------------------


def _call(llm: Callable[[str], str], prompt: str) -> Optional[str]:
    """Call ``llm`` and return a usable string, or ``None``."""
    reply = llm(prompt)
    if reply is None:
        return None
    if not isinstance(reply, str):
        reply = str(reply)
    reply = reply.strip()
    return reply or None


def _unquote(text: str) -> str:
    """Drop wrapping quotes or a markdown fence an LLM may have added."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _summary_prompt(
    summary_text: str,
    decisions: Sequence[Decision],
    actions: Sequence[Action],
) -> str:
    """Prompt asking for a tighter summary of facts already extracted."""
    lines = [
        "Rewrite the meeting notes below as a single clear paragraph of at most",
        "four sentences. Use only what the notes contain: do not invent names,",
        "dates, decisions or tasks, and do not add commentary. Reply with the",
        "paragraph only.",
        "",
        "NOTES:",
        summary_text,
    ]
    if decisions:
        lines.append("")
        lines.append("DECISIONS:")
        for decision in list(decisions)[:10]:
            lines.append("- " + truncate(decision.text, 160))
    if actions:
        lines.append("")
        lines.append("ACTION ITEMS:")
        for action in list(actions)[:10]:
            lines.append("- " + truncate(action.describe(), 160))
    return "\n".join(lines)


def _action_prompt(actions: Sequence[Action]) -> str:
    """Prompt asking for the same tasks, phrased as task lines."""
    lines = [
        "Rewrite each numbered meeting action item below as one short task line,",
        "starting with a verb. Keep the same meaning, the same owner and the same",
        "date. Do not merge, drop or add items. Reply with one numbered line per",
        "input, in the same order, and nothing else.",
        "",
    ]
    for number, action in enumerate(actions, start=1):
        lines.append("%d. %s" % (number, truncate(action.text, 200)))
    return "\n".join(lines)


def _parse_numbered(reply: str, count: int) -> Dict[int, str]:
    """Map numbered reply lines back onto action positions."""
    rewritten = {}  # type: Dict[int, str]
    for line in _unquote(reply).splitlines():
        match = _NUMBERED_LINE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        text = match.group(2).strip().strip("*").strip()
        if not 1 <= number <= count:
            continue
        if not text or len(text) > _MAX_LLM_ACTION_CHARS:
            continue
        rewritten[number - 1] = text
    return rewritten


def polish_with_llm(
    llm: Callable[[str], str],
    summary_text: str,
    decisions: Sequence[Decision],
    actions: Sequence[Action],
) -> Tuple[Optional[str], Dict[int, str], List[str]]:
    """Ask ``llm`` to improve the summary and the action phrasing.

    Returns ``(summary_or_None, {action_index: new_text}, warnings)``. Any
    exception raised by ``llm``, and any reply that cannot be used, becomes a
    warning and leaves the extractive result in place. This function never lets
    a caller's callable break the report.
    """
    warnings = []  # type: List[str]
    if not callable(llm):
        return None, {}, ["llm must be a callable taking a prompt and returning text; ignored"]

    new_summary = None  # type: Optional[str]
    try:
        reply = _call(llm, _summary_prompt(summary_text, decisions, actions))
    except Exception as error:  # noqa: BLE001 - a caller's llm may raise anything
        logger.warning("llm summary call failed: %s", error)
        return (
            None,
            {},
            ["llm call failed (%s: %s); report built without it" % (type(error).__name__, error)],
        )
    if reply is None:
        warnings.append("llm returned nothing for the summary; kept the extractive one")
    else:
        candidate = _unquote(reply)
        if not candidate or len(candidate) > _MAX_LLM_SUMMARY_CHARS:
            warnings.append("llm summary was empty or too long; kept the extractive one")
        else:
            new_summary = " ".join(candidate.split())

    rewritten = {}  # type: Dict[int, str]
    subset = list(actions)[:_MAX_LLM_ACTIONS]
    if subset:
        try:
            reply = _call(llm, _action_prompt(subset))
        except Exception as error:  # noqa: BLE001 - a caller's llm may raise anything
            logger.warning("llm action call failed: %s", error)
            warnings.append(
                "llm action rewrite failed (%s: %s); kept the extracted wording"
                % (type(error).__name__, error)
            )
            reply = None
        if reply is not None:
            rewritten = _parse_numbered(reply, len(subset))
            if not rewritten:
                warnings.append("llm action rewrite was unusable; kept the extracted wording")

    return new_summary, rewritten, warnings
