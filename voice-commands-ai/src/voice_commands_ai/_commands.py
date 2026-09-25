"""The command registry: add patterns, match text, run handlers."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from ._match import Match
from ._matcher import Alignment, align, describe_words, ignored_phrases, prepare
from ._pattern import Pattern, parse_pattern
from ._text import filler_table

logger = logging.getLogger(__name__)

#: Two candidates whose confidences are this close count as a tie, and the
#: more specific pattern wins.
SPECIFICITY_MARGIN = 0.05


class CommandError(RuntimeError):
    """A command's handler raised. The original exception is ``__cause__``.

    Attributes: ``name`` (the command), ``match`` (what was matched) and
    ``original`` (the exception the handler raised).
    """

    def __init__(self, name: str, match: Match, original: BaseException) -> None:
        super().__init__(
            "command '%s' failed on %r: %s: %s"
            % (name, match.text, type(original).__name__, original)
        )
        self.name = name
        self.match = match
        self.original = original


@dataclass(frozen=True)
class Command:
    """One registered command."""

    name: str
    pattern: str
    handler: Optional[Callable[..., Any]] = field(default=None, compare=False)
    examples: Tuple[str, ...] = ()
    slots: Tuple[Tuple[str, str], ...] = ()  # (slot name, slot type)
    parsed: Pattern = field(default=None, repr=False, compare=False)  # type: ignore[assignment]


def _default_name(pattern: str, handler: Optional[Callable[..., Any]]) -> str:
    name = getattr(handler, "__name__", "") if handler is not None else ""
    if name and name.isidentifier():
        return name
    return pattern


def _check_handler(handler: Callable[..., Any], name: str, slot_names: Sequence[str]) -> None:
    """Fail at registration, not at run time, if the handler cannot take the slots."""
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):  # builtins and some C callables have no signature
        return
    try:
        signature.bind(**{slot: None for slot in slot_names})
    except TypeError as exc:
        wanted = ", ".join(slot_names) if slot_names else "no arguments"
        raise ValueError(
            "the handler for command '%s' cannot be called with the pattern's slots (%s): %s"
            % (name, wanted, exc)
        ) from None


class Commands:
    """A set of voice commands: patterns with slots, matched fuzzily against text.

    >>> cmds = Commands()
    >>> _ = cmds.add("set temperature to {value:number} degrees", name="thermostat")
    >>> cmds.match("um set the temperature to twenty one degrees please").slots
    {'value': 21}

    ``min_confidence`` is the floor below which :meth:`match` returns None
    rather than a poor guess. ``fillers`` adds phrases that may be skipped
    for free (on top of "please", "could you", "um" and the rest).
    """

    def __init__(self, *, min_confidence: float = 0.6, fillers: Iterable[str] = ()) -> None:
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise TypeError("min_confidence must be a number between 0 and 1")
        if not 0.0 < float(min_confidence) <= 1.0:
            raise ValueError("min_confidence must be above 0 and at most 1, got %r" % (min_confidence,))
        if isinstance(fillers, str):
            fillers = [fillers]
        self.min_confidence = float(min_confidence)
        self._fillers = filler_table(fillers)
        self._commands: List[Command] = []

    # ------------------------------------------------------------ registry

    def add(
        self,
        pattern: str,
        handler: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        examples: Optional[Iterable[str]] = None,
    ) -> "Commands":
        """Register a command and return ``self`` so calls can be chained.

        ``pattern`` is a phrase with slots, e.g. ``"turn {state:on|off} the
        {device:text}"``. ``handler`` (optional) is called by :meth:`run` with
        the slots as keyword arguments. ``name`` defaults to the handler's
        function name, else the pattern. Each of ``examples`` must match this
        pattern, which catches typos in a pattern at start-up.
        """
        parsed = parse_pattern(pattern)
        if handler is not None and not callable(handler):
            raise TypeError("handler must be callable or None, got %s" % type(handler).__name__)
        if name is None:
            name = _default_name(pattern, handler)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("a command name must be a non-empty string")
        if any(c.name == name for c in self._commands):
            raise ValueError("a command named '%s' is already registered" % name)
        if handler is not None:
            _check_handler(handler, name, parsed.slot_names)
        if isinstance(examples, str):
            examples = [examples]
        example_list = tuple(examples or ())
        command = Command(name, pattern, handler, example_list, parsed.slots, parsed)
        for example in example_list:
            if not isinstance(example, str):
                raise TypeError("examples must be strings, got %r" % (example,))
            found = self._to_match(align(parsed, prepare(example, self._fillers)), command)
            if found.confidence < self.min_confidence:
                raise ValueError(
                    "example %r does not match its own pattern %r (confidence %.2f, floor %.2f)"
                    % (example, pattern, found.confidence, self.min_confidence)
                )
        if not parsed.has_fixed_words:
            logger.warning(
                "pattern %r has no fixed words, so it will match almost anything", pattern
            )
        self._commands.append(command)
        return self

    def __len__(self) -> int:
        return len(self._commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(list(self._commands))

    def __contains__(self, name: object) -> bool:
        return any(c.name == name for c in self._commands)

    def __repr__(self) -> str:
        return "Commands(%d command%s, min_confidence=%.2f)" % (
            len(self._commands), "" if len(self._commands) == 1 else "s", self.min_confidence
        )

    # ------------------------------------------------------------ matching

    def _to_match(self, alignment: Alignment, command: Command) -> Match:
        utt = alignment.utterance
        slots: Dict[str, Any] = {}
        slot_words: Dict[str, str] = {}
        corrections: List[Dict[str, str]] = []
        missing: List[str] = []
        reordered: List[str] = []
        seen_ranges = set()
        for res in alignment.results:
            element = res.element
            if res.status == "missing":
                missing.append(element.label)
                continue
            if not res.found:
                continue
            if res.status == "reordered":
                reordered.append(element.label)
            heard = describe_words(utt, res.start, res.end)
            if element.is_slot:
                slots[element.text] = res.value
                slot_words[element.text] = res.value if element.kind == "text" else heard
                if element.kind == "choice" and res.score < 1.0:
                    corrections.append({"heard": heard, "as": res.value, "how": res.how})
            elif res.how != "exact" and (res.start, res.end) not in seen_ranges:
                seen_ranges.add((res.start, res.end))
                expected = " ".join(
                    r.element.text for r in alignment.results
                    if r.found and (r.start, r.end) == (res.start, res.end) and not r.element.is_slot
                )
                corrections.append({"heard": heard, "as": expected, "how": res.how})
        extra = [utt.tokens[i].surface for i in alignment.leftovers if not utt.tokens[i].is_filler]
        replaced = [
            {"expected": res.element.label, "heard": " ".join(utt.tokens[i].surface for i in where)}
            for res, where in alignment.replaced
        ]
        return Match(
            name=command.name,
            slots=slots,
            confidence=alignment.confidence,
            pattern=command.pattern,
            text=utt.original,
            slot_words=slot_words,
            corrections=corrections,
            ignored=ignored_phrases(alignment),
            extra=extra,
            missing=missing,
            reordered=reordered,
            replaced=replaced,
            coverage=alignment.coverage,
            precision=alignment.precision,
            specificity=command.parsed.specificity,
            threshold=self.min_confidence,
            _command=command,
        )

    def _ranked(self, text: str) -> Tuple[List[Match], Optional[Match]]:
        """All candidates, winner first (when there is one)."""
        if not isinstance(text, str):
            raise TypeError("text must be a str, got %s" % type(text).__name__)
        if not self._commands or not text.strip():
            return [], None
        utt = prepare(text, self._fillers)
        if not utt.tokens:
            return [], None
        candidates = [self._to_match(align(c.parsed, utt), c) for c in self._commands]
        order = sorted(range(len(candidates)), key=lambda k: -candidates[k].confidence)
        ranked = [candidates[k] for k in order]
        position = {id(c): n for n, c in enumerate(ranked)}
        passing = [c for c in ranked if c.confidence >= self.min_confidence]
        if not passing:
            return ranked, None
        top = passing[0].confidence
        contenders = [c for c in passing if c.confidence >= top - SPECIFICITY_MARGIN]
        winner = max(
            contenders, key=lambda c: (c.specificity, c.confidence, -position[id(c)])
        )
        if winner is not passing[0]:
            beaten = [c for c in contenders if c is not winner and c.confidence >= winner.confidence]
            if beaten:
                winner.chosen_because = (
                    "its pattern is more specific than %s, which scored as high or higher"
                    % ", ".join("'%s' (%.2f)" % (c.name, c.confidence) for c in beaten)
                )
        ranked.pop(position[id(winner)])
        ranked.insert(0, winner)
        return ranked, winner

    def match(self, text: str) -> Optional[Match]:
        """The best command for ``text``, or None if nothing clears the confidence floor.

        Empty text, text with no words, and text that matches nothing well all
        return None. When several commands clear the floor, the best one is
        returned and the others are listed in ``Match.alternatives``.
        """
        ranked, winner = self._ranked(text)
        if winner is None:
            if ranked:
                logger.debug(
                    "no command for %r; closest was %r at %.2f (floor %.2f)",
                    text, ranked[0].name, ranked[0].confidence, self.min_confidence,
                )
            return None
        winner.alternatives = [
            c for c in ranked[1:] if c.confidence >= self.min_confidence
        ]
        logger.debug("matched %r to %r at %.2f", text, winner.name, winner.confidence)
        return winner

    def rank(self, text: str) -> List[Match]:
        """Every command scored against ``text``, best first, including those
        below the floor. Use it to see why something did or did not match."""
        ranked, _ = self._ranked(text)
        return ranked

    # ------------------------------------------------------------ actions

    def run(self, text: Union[str, Match, None], *, default: Any = None) -> Any:
        """Match ``text`` and call the command's handler with the slots as keyword arguments.

        Also accepts a :class:`Match` (for example from :meth:`listen`). Returns
        the handler's result; the ``Match`` itself for a command registered
        without a handler; ``default`` when nothing matched. If the handler
        raises, a :class:`CommandError` naming the command is raised from it.
        """
        if text is None:
            return default
        if isinstance(text, Match):
            found: Optional[Match] = text
            command = next(
                (c for c in self._commands if text.command is not None and c is text.command),
                None,
            )
            if command is None:
                command = next(
                    (c for c in self._commands if c.name == text.name and c.pattern == text.pattern),
                    None,
                )
            if command is None:
                raise ValueError("match for '%s' does not belong to these commands" % text.name)
        else:
            found = self.match(text)
            if found is None:
                return default
            command = found.command
        assert found is not None and command is not None
        if command.handler is None:
            return found
        try:
            return command.handler(**found.slots)
        except Exception as exc:
            logger.debug("handler for %r raised %r", command.name, exc)
            raise CommandError(command.name, found, exc) from exc

    def listen(self, transcribe: Callable[[Any], Any], audio: Any) -> Optional[Match]:
        """Transcribe ``audio`` with your own ``transcribe(audio) -> str`` and match the text.

        This package does no speech recognition itself. ``transcribe`` may also
        return a list of alternative transcripts (best first); the one that
        matches best wins, earlier ones winning ties. A transcriber that returns
        None or "" means nothing was heard, and gives None. Errors raised by the
        transcriber propagate unchanged.
        """
        if not callable(transcribe):
            raise TypeError("transcribe must be a callable taking the audio and returning text")
        heard = transcribe(audio)
        if heard is None:
            return None
        if isinstance(heard, str):
            return self.match(heard)
        if isinstance(heard, (list, tuple)):
            best: Optional[Match] = None
            for option in heard:
                if not isinstance(option, str):
                    raise TypeError(
                        "transcribe(audio) returned a list containing %s; expected strings"
                        % type(option).__name__
                    )
                found = self.match(option)
                if found is not None and (best is None or found.confidence > best.confidence):
                    best = found
            return best
        raise TypeError(
            "transcribe(audio) must return a str (or a list of alternative transcripts), got %s"
            % type(heard).__name__
        )
