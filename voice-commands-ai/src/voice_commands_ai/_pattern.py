"""Command patterns: "set temperature to {value:number} degrees".

Syntax
------
* plain words are matched fuzzily, ignoring case, accents and punctuation;
* ``[words]`` are optional words ("turn on [the] lights");
* ``{name}`` or ``{name:word}`` is exactly one word;
* ``{name:number}`` is a spoken or written number ("twenty one", "21", "2.5");
* ``{name:on|off}`` is one of the listed choices (a choice may be several words:
  ``{room:living room|kitchen}``);
* ``{name:text}`` is free text: one or more words, up to the next fixed word of
  the pattern, or to the end of what was said when it comes last.
"""

from __future__ import annotations

import keyword
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

from ._text import FUNCTION_WORDS, raw_tokens

SLOT_TYPES = ("number", "word", "text")

# Evidence weights. A missing content word costs more than a missing "the";
# a free-text slot proves little because it accepts anything.
_WEIGHT = {"number": 1.0, "choice": 1.0, "word": 0.75, "text": 0.5}
# Specificity decides between two patterns that both match well: fixed words
# beat constrained slots, which beat open ones.
_SPECIFICITY = {"number": 0.6, "choice": 0.6, "word": 0.3, "text": 0.1, "optional": 0.2}

_PIECE_RE = re.compile(r"\{([^{}]*)\}|\[([^\[\]{}]*)\]|([^{}\[\]]+)|(.)", re.S)


@dataclass(frozen=True)
class Element:
    """One part of a pattern: a fixed word, an optional word or a slot."""

    kind: str  # "literal", "optional", "number", "word", "text" or "choice"
    text: str  # the word as written, or the slot name
    key: str = ""  # comparison key of a literal
    options: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()  # choice: (as written, keys)

    @property
    def is_slot(self) -> bool:
        """True for slots, False for fixed and optional words."""
        return self.kind in ("number", "word", "text", "choice")

    @property
    def required(self) -> bool:
        """Optional words are the only elements that may be absent for free."""
        return self.kind != "optional"

    @property
    def weight(self) -> float:
        """Share of the confidence this element carries when found."""
        if self.kind == "optional":
            return 0.0
        if self.kind == "literal":
            return 0.5 if self.key in FUNCTION_WORDS else 1.0
        return _WEIGHT[self.kind]

    @property
    def specificity(self) -> float:
        """How much this element narrows down what the pattern accepts."""
        if self.kind == "literal":
            return 0.5 if self.key in FUNCTION_WORDS else 1.0
        return _SPECIFICITY[self.kind]

    @property
    def label(self) -> str:
        """How the element reads in explanations: a word, or ``{name}``."""
        return "{%s}" % self.text if self.is_slot else self.text


@dataclass(frozen=True)
class Pattern:
    """A parsed pattern: its source text, elements and slot types."""

    source: str
    elements: Tuple[Element, ...]
    slots: Tuple[Tuple[str, str], ...] = field(default=())  # (name, type)

    @property
    def slot_names(self) -> Tuple[str, ...]:
        """Slot names in the order they appear."""
        return tuple(name for name, _ in self.slots)

    @property
    def specificity(self) -> float:
        """Sum of element specificities; higher means the pattern accepts less."""
        return sum(e.specificity for e in self.elements)

    @property
    def has_fixed_words(self) -> bool:
        """True when at least one required word is fixed rather than a slot."""
        return any(e.kind == "literal" for e in self.elements)


def _words(text: str) -> List[Tuple[str, str]]:
    """(surface, key) for each word of a pattern fragment."""
    return [(t.surface, t.key) for t in raw_tokens(unicodedata.normalize("NFC", text))]


def _slot(body: str, pattern: str) -> Element:
    name, sep, spec = body.partition(":")
    name = name.strip()
    if not name:
        raise ValueError("pattern %r has a slot with no name: {%s}" % (pattern, body))
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            "slot name %r in pattern %r must be a valid Python identifier, because slots "
            "are passed to the handler as keyword arguments" % (name, pattern)
        )
    if not sep:
        return Element("word", name)
    spec = spec.strip()
    if "|" in spec:
        options = []
        seen = set()
        for raw in spec.split("|"):
            words = _words(raw)
            if not words:
                raise ValueError(
                    "slot {%s} in pattern %r has an empty choice; write options as a|b|c"
                    % (name, pattern)
                )
            keys = tuple(k for _, k in words)
            if keys in seen:
                raise ValueError(
                    "slot {%s} in pattern %r lists the choice %r twice" % (name, pattern, raw.strip())
                )
            seen.add(keys)
            options.append((" ".join(raw.split()), keys))
        return Element("choice", name, options=tuple(options))
    kind = spec.lower()
    if kind not in SLOT_TYPES:
        raise ValueError(
            "slot {%s} in pattern %r has unknown type %r; use number, word, text, or a|b|c "
            "for a choice" % (name, pattern, spec)
        )
    return Element(kind, name)


def parse_pattern(pattern: str) -> Pattern:
    """Parse a pattern string, raising ``ValueError`` with the reason if it is invalid."""
    if not isinstance(pattern, str):
        raise TypeError("a pattern must be a str, got %s" % type(pattern).__name__)
    if not pattern.strip():
        raise ValueError("a pattern cannot be empty")
    elements: List[Element] = []
    for m in _PIECE_RE.finditer(pattern):
        slot, optional, plain, stray = m.groups()
        if stray is not None:
            raise ValueError(
                "pattern %r has an unmatched %r at position %d" % (pattern, stray, m.start())
            )
        if slot is not None:
            elements.append(_slot(slot, pattern))
        elif optional is not None:
            words = _words(optional)
            if not words:
                raise ValueError("pattern %r has an empty optional group []" % (pattern,))
            elements.extend(Element("optional", s, k) for s, k in words)
        else:
            elements.extend(Element("literal", s, k) for s, k in _words(plain))
    if not any(e.required for e in elements):
        raise ValueError(
            "pattern %r needs at least one word or slot that is not optional" % (pattern,)
        )
    names = [e.text for e in elements if e.is_slot]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            "pattern %r uses the slot name(s) %s more than once" % (pattern, ", ".join(duplicates))
        )
    required = [e for e in elements if e.required]
    for a, b in zip(required, required[1:]):
        if a.kind == "text" and b.kind == "text":
            raise ValueError(
                "pattern %r has two text slots in a row ({%s} {%s}); put a fixed word "
                "between them so they can be told apart" % (pattern, a.text, b.text)
            )
    slots = tuple(
        (e.text, "choice" if e.kind == "choice" else e.kind) for e in elements if e.is_slot
    )
    return Pattern(pattern, tuple(elements), slots)
