"""The result of matching: which command, which slot values, and why."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Match:
    """A command recognised in some text, with its slot values and the evidence.

    ``name``, ``slots``, ``confidence`` and ``pattern`` answer "what was asked";
    the remaining fields explain how the text was read, so a surprising match
    can be understood without re-running anything. ``summary()`` prints all of
    it; ``to_dict()`` gives the same as JSON-safe data.
    """

    name: str
    slots: Dict[str, Any]
    confidence: float
    pattern: str
    text: str = ""
    #: Other commands that also cleared the confidence floor, best first.
    alternatives: List["Match"] = field(default_factory=list)
    #: The words each slot value came from, as heard.
    slot_words: Dict[str, str] = field(default_factory=dict)
    #: Words taken as a pattern word they did not exactly spell:
    #: ``{"heard": ..., "as": ..., "how": ...}``.
    corrections: List[Dict[str, str]] = field(default_factory=list)
    #: Filler words and hesitations that were skipped.
    ignored: List[str] = field(default_factory=list)
    #: Words the pattern does not explain.
    extra: List[str] = field(default_factory=list)
    #: Pattern words and slots that were not found.
    missing: List[str] = field(default_factory=list)
    #: Pattern words and slots that were found, but out of order.
    reordered: List[str] = field(default_factory=list)
    #: Pattern words with a different word said in their place:
    #: ``{"expected": ..., "heard": ...}``.
    replaced: List[Dict[str, str]] = field(default_factory=list)
    #: Weighted share of the pattern that was found (0-1).
    coverage: float = 0.0
    #: Weighted share of the words that the pattern explains (0-1).
    precision: float = 0.0
    #: How narrowly the pattern is written; breaks near-ties between commands.
    specificity: float = 0.0
    #: The confidence floor that was in force.
    threshold: float = 0.0
    #: Why this command was chosen over an alternative that scored as high.
    chosen_because: str = ""
    _command: Any = field(default=None, repr=False, compare=False)

    def __bool__(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "Match(name=%r, slots=%r, confidence=%.3f)" % (self.name, self.slots, self.confidence)

    def summary(self) -> str:
        """A plain-text explanation of the match, one fact per line."""
        lines = [
            "Matched command '%s' with confidence %.2f (floor %.2f)"
            % (self.name, self.confidence, self.threshold),
            'Heard:   "%s"' % " ".join(self.text.split()),
            "Pattern: %s" % self.pattern,
        ]
        if self.slots:
            lines.append("Slots:")
            for key, value in self.slots.items():
                words = self.slot_words.get(key, "")
                shown = repr(value) if isinstance(value, str) else str(value)
                if words and words != value:
                    lines.append('  %s = %s  (from "%s")' % (key, shown, words))
                else:
                    lines.append("  %s = %s" % (key, shown))
        else:
            lines.append("Slots: none")
        lines.append(
            "Scores: %.0f%% of the pattern found, %.0f%% of the words explained"
            % (100 * self.coverage, 100 * self.precision)
        )
        if self.corrections:
            lines.append(
                "Corrections: "
                + "; ".join('"%s" taken as "%s" (%s)' % (c["heard"], c["as"], c["how"])
                            for c in self.corrections)
            )
        if self.reordered:
            lines.append("Out of order: " + ", ".join(self.reordered))
        if self.missing:
            lines.append("Not heard: " + ", ".join(self.missing))
        if self.replaced:
            lines.append(
                "Said instead: "
                + "; ".join('"%s" where "%s" was expected' % (r["heard"], r["expected"])
                            for r in self.replaced)
            )
        if self.extra:
            lines.append("Extra words: " + ", ".join(self.extra))
        if self.ignored:
            lines.append("Ignored filler: " + ", ".join(self.ignored))
        if self.chosen_because:
            lines.append("Chosen because: " + self.chosen_because)
        if self.alternatives:
            lines.append(
                "Also matched: "
                + ", ".join("'%s' (%.2f)" % (a.name, a.confidence) for a in self.alternatives)
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def to_dict(self, _nested: bool = False) -> Dict[str, Any]:
        """JSON-safe dict of everything in the match (alternatives one level deep)."""
        data: Dict[str, Any] = {
            "name": self.name,
            "slots": dict(self.slots),
            "confidence": round(self.confidence, 4),
            "pattern": self.pattern,
            "text": self.text,
            "slot_words": dict(self.slot_words),
            "corrections": [dict(c) for c in self.corrections],
            "ignored": list(self.ignored),
            "extra": list(self.extra),
            "missing": list(self.missing),
            "reordered": list(self.reordered),
            "replaced": [dict(r) for r in self.replaced],
            "coverage": round(self.coverage, 4),
            "precision": round(self.precision, 4),
            "specificity": round(self.specificity, 4),
            "threshold": self.threshold,
            "chosen_because": self.chosen_because,
        }
        if not _nested:
            data["alternatives"] = [a.to_dict(_nested=True) for a in self.alternatives]
        return data

    @property
    def command(self) -> Optional[Any]:
        """The registered command this match belongs to (name, pattern, handler, examples)."""
        return self._command
