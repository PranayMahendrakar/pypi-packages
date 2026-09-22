"""The result object: a meeting report that explains itself.

:class:`MeetingReport` is a plain dataclass. Everything on it is a list of
dataclasses or a number, so it prints, serialises and diffs without surprises.
Three renderings are provided: :meth:`MeetingReport.summary` for a person
reading a terminal, :meth:`MeetingReport.to_dict` for JSON, and
:meth:`MeetingReport.to_markdown` for minutes that get pasted somewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._extract import Action, Decision, Question
from ._parse import Turn, TurnList
from ._participation import Participation
from ._summarise import format_duration
from ._topics import Topic

__all__ = ["MeetingReport"]

_RULE = "=" * 62
_THIN = "-" * 62


def _stamp(seconds: Optional[float]) -> str:
    """``00:01:02`` for a start time, or an empty string when untimed."""
    if seconds is None:
        return ""
    total = int(round(float(seconds)))
    hours, rest = divmod(max(0, total), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%02d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)


@dataclass
class MeetingReport:
    """What a transcript turned out to contain.

    Attributes
    ----------
    summary_text:
        A short paragraph describing the meeting. Extractive by default; replaced
        by the LLM's wording when an ``llm`` callable was supplied and worked.
    decisions:
        :class:`~meeting_intelligence.Decision` records, each carrying the cue
        phrase that matched.
    action_items:
        :class:`~meeting_intelligence.Action` records with ``text``, ``owner``,
        ``due`` and ``confidence``.
    questions:
        Every question asked, with whether a later turn answered it.
    topics:
        Segments of the discussion, each labelled from its distinctive words.
    participation:
        One row per speaker, heaviest speaker first.
    duration:
        Seconds from the first to the last timestamp, or ``None`` when the
        transcript carried no timings.
    """

    summary_text: str = ""
    decisions: List[Decision] = field(default_factory=list)
    action_items: List[Action] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    topics: List[Topic] = field(default_factory=list)
    participation: List[Participation] = field(default_factory=list)
    duration: Optional[float] = None

    turns: List[Turn] = field(default_factory=TurnList)
    speakers: List[str] = field(default_factory=list)
    source_format: str = "empty"
    n_turns: int = 0
    n_words: int = 0
    interruptions_basis: str = "none"
    llm_used: bool = False
    warnings: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------- views

    @property
    def is_empty(self) -> bool:
        """True when the transcript held nothing to report on."""
        return self.n_turns == 0

    @property
    def open_questions(self) -> List[Question]:
        """Questions that nobody answered later in the meeting."""
        return [question for question in self.questions if not question.answered]

    @property
    def answered_questions(self) -> List[Question]:
        """Questions a later turn answered."""
        return [question for question in self.questions if question.answered]

    @property
    def owned_actions(self) -> List[Action]:
        """Action items whose owner could be resolved."""
        return [action for action in self.action_items if action.owner]

    @property
    def unowned_actions(self) -> List[Action]:
        """Action items still waiting for a name."""
        return [action for action in self.action_items if not action.owner]

    @property
    def duration_text(self) -> Optional[str]:
        """Human wording of :attr:`duration`, or ``None`` when untimed."""
        return format_duration(self.duration)

    def actions_for(self, owner: str) -> List[Action]:
        """Action items belonging to ``owner``, matched case-insensitively."""
        needle = str(owner).strip().lower()
        return [
            action
            for action in self.action_items
            if action.owner and action.owner.lower() == needle
        ]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            "MeetingReport(turns=%d, speakers=%d, decisions=%d, actions=%d, "
            "open_questions=%d, topics=%d)"
            % (
                self.n_turns,
                len(self.speakers),
                len(self.decisions),
                len(self.action_items),
                len(self.open_questions),
                len(self.topics),
            )
        )

    # ------------------------------------------------------------ renderings

    def summary(self) -> str:
        """The whole report as plain text, ASCII punctuation only."""
        lines = ["Meeting report", _RULE]
        if self.is_empty:
            lines.append("Empty transcript: nothing to report.")
            for warning in self.warnings:
                lines.append("Note: " + warning)
            return "\n".join(lines)

        lines.append(self.summary_text)
        lines.append("")
        header = "%d turns, %d words, %d speaker(s), format %s" % (
            self.n_turns,
            self.n_words,
            len(self.speakers),
            self.source_format,
        )
        if self.duration_text:
            header += ", about %s" % self.duration_text
        if self.llm_used:
            header += ", summary polished by the supplied llm"
        lines.append(header)

        lines.append("")
        lines.append("Decisions (%d)" % len(self.decisions))
        lines.append(_THIN)
        if self.decisions:
            for decision in self.decisions:
                who = " -- %s" % decision.speaker if decision.speaker else ""
                lines.append("  [%.2f] %s%s" % (decision.confidence, decision.text, who))
                lines.append("         cue: %s" % decision.cue)
        else:
            lines.append("  none found")

        lines.append("")
        lines.append("Action items (%d)" % len(self.action_items))
        lines.append(_THIN)
        if self.action_items:
            for action in self.action_items:
                lines.append("  [%.2f] %s" % (action.confidence, action.describe()))
                lines.append("         cue: %s" % action.cue)
        else:
            lines.append("  none found")

        open_questions = self.open_questions
        lines.append("")
        lines.append(
            "Questions (%d asked, %d still open)" % (len(self.questions), len(open_questions))
        )
        lines.append(_THIN)
        if open_questions:
            for question in open_questions:
                who = " -- %s" % question.speaker if question.speaker else ""
                lines.append("  open: %s%s" % (question.text, who))
        for question in self.answered_questions:
            lines.append("  answered: %s" % question.text)
            if question.answer_text:
                by = " (%s)" % question.answered_by if question.answered_by else ""
                lines.append("            -> %s%s" % (question.answer_text, by))
        if not self.questions:
            lines.append("  none asked")

        lines.append("")
        lines.append("Topics (%d)" % len(self.topics))
        lines.append(_THIN)
        for topic in self.topics:
            span = "turns %d-%d" % (topic.start_turn, topic.end_turn)
            if topic.start is not None:
                span += ", from %s" % _stamp(topic.start)
            lines.append("  %s (%s)" % (topic.label, span))
            if topic.preview:
                lines.append("    %s" % topic.preview)
        if not self.topics:
            lines.append("  none")

        lines.append("")
        lines.append("Participation (interruptions from %s)" % self.interruptions_basis)
        lines.append(_THIN)
        for row in self.participation:
            lines.append("  " + row.describe())
        if not self.participation:
            lines.append("  none")

        if self.warnings:
            lines.append("")
            lines.append("Notes")
            lines.append(_THIN)
            for warning in self.warnings:
                lines.append("  - %s" % warning)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the whole report."""
        return {
            "summary_text": self.summary_text,
            "decisions": [item.to_dict() for item in self.decisions],
            "action_items": [item.to_dict() for item in self.action_items],
            "questions": [item.to_dict() for item in self.questions],
            "topics": [item.to_dict() for item in self.topics],
            "participation": [item.to_dict() for item in self.participation],
            "duration": self.duration,
            "duration_text": self.duration_text,
            "speakers": list(self.speakers),
            "source_format": self.source_format,
            "n_turns": self.n_turns,
            "n_words": self.n_words,
            "n_decisions": len(self.decisions),
            "n_action_items": len(self.action_items),
            "n_open_questions": len(self.open_questions),
            "interruptions_basis": self.interruptions_basis,
            "llm_used": self.llm_used,
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """The report as Markdown, ready to paste into minutes."""
        lines = ["# Meeting report", ""]
        if self.is_empty:
            lines.append("_Empty transcript: nothing to report._")
            if self.warnings:
                lines.append("")
                for warning in self.warnings:
                    lines.append("- %s" % warning)
            return "\n".join(lines) + "\n"

        lines.append(self.summary_text)
        lines.append("")
        facts = ["%d turns" % self.n_turns, "%d words" % self.n_words]
        if self.speakers:
            facts.append("%d speakers" % len(self.speakers))
        if self.duration_text:
            facts.append("about %s" % self.duration_text)
        facts.append("source format `%s`" % self.source_format)
        lines.append("*%s*" % " - ".join(facts))

        lines.append("")
        lines.append("## Decisions")
        lines.append("")
        if self.decisions:
            for decision in self.decisions:
                who = " -- %s" % decision.speaker if decision.speaker else ""
                lines.append(
                    "- %s%s  \n  <sub>cue `%s`, confidence %.2f</sub>"
                    % (decision.text, who, decision.cue, decision.confidence)
                )
        else:
            lines.append("_None found._")

        lines.append("")
        lines.append("## Action items")
        lines.append("")
        if self.action_items:
            lines.append("| Owner | Task | Due | Confidence |")
            lines.append("| --- | --- | --- | --- |")
            for action in self.action_items:
                lines.append(
                    "| %s | %s | %s | %.2f |"
                    % (
                        action.owner or "_unassigned_",
                        action.text.replace("|", "\\|"),
                        action.due or "-",
                        action.confidence,
                    )
                )
        else:
            lines.append("_None found._")

        lines.append("")
        lines.append("## Open questions")
        lines.append("")
        open_questions = self.open_questions
        if open_questions:
            for question in open_questions:
                who = " -- %s" % question.speaker if question.speaker else ""
                lines.append("- %s%s" % (question.text, who))
        else:
            lines.append("_None left open._")

        lines.append("")
        lines.append("## Topics")
        lines.append("")
        if self.topics:
            for topic in self.topics:
                span = "turns %d-%d" % (topic.start_turn, topic.end_turn)
                if topic.start is not None:
                    span += ", from %s" % _stamp(topic.start)
                lines.append("- **%s** (%s)" % (topic.label, span))
        else:
            lines.append("_None._")

        lines.append("")
        lines.append("## Participation")
        lines.append("")
        if self.participation:
            lines.append("| Speaker | Turns | Words | Share | Longest run | Interruptions |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for row in self.participation:
                lines.append(
                    "| %s | %d | %d | %.0f%% | %d | %d |"
                    % (
                        row.speaker,
                        row.turns,
                        row.words,
                        row.share * 100.0,
                        row.longest_monologue,
                        row.interruptions,
                    )
                )
        else:
            lines.append("_None._")

        if self.warnings:
            lines.append("")
            lines.append("## Notes")
            lines.append("")
            for warning in self.warnings:
                lines.append("- %s" % warning)
        return "\n".join(lines) + "\n"
