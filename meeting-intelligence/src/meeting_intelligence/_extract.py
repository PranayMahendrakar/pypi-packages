"""Cue-phrase extraction of decisions, action items and questions.

Nothing here learns anything. Every match comes from a phrase list that is
published as part of the API -- :data:`DECISION_CUES`, :data:`ACTION_RULES` and
:data:`DUE_PATTERNS` -- so a result can always be traced back to the phrase that
produced it, and the lists can be read, audited or extended by the caller.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._parse import Turn
from ._text import STOP_WORDS, content_words, fold, tokenize, word_count

__all__ = [
    "Action",
    "Decision",
    "Question",
    "DECISION_CUES",
    "ACTION_RULES",
    "DUE_PATTERNS",
    "extract_decisions",
    "extract_actions",
    "extract_questions",
    "find_due",
    "unanswered",
]

# --------------------------------------------------------------------------
# Result records
# --------------------------------------------------------------------------


@dataclass
class Decision:
    """Something the meeting settled, and the phrase that gave it away."""

    text: str
    speaker: Optional[str] = None
    cue: str = ""
    confidence: float = 0.5
    turn_index: int = -1
    start: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the decision."""
        return {
            "text": self.text,
            "speaker": self.speaker,
            "cue": self.cue,
            "confidence": round(self.confidence, 3),
            "turn_index": self.turn_index,
            "start": self.start,
        }


@dataclass
class Action:
    """A task someone took on.

    Constructed positionally as ``Action(text, owner, due, confidence)``; the
    provenance fields after those carry the phrase and turn it came from.
    ``source_text`` holds the original sentence when an ``llm`` callable
    rewrote ``text``, and is ``None`` otherwise.
    """

    text: str
    owner: Optional[str] = None
    due: Optional[str] = None
    confidence: float = 0.5
    speaker: Optional[str] = None
    cue: str = ""
    turn_index: int = -1
    start: Optional[float] = None
    owner_is_speaker: bool = False
    source_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the action item."""
        return {
            "text": self.text,
            "owner": self.owner,
            "due": self.due,
            "confidence": round(self.confidence, 3),
            "speaker": self.speaker,
            "cue": self.cue,
            "turn_index": self.turn_index,
            "start": self.start,
            "owner_is_speaker": self.owner_is_speaker,
            "source_text": self.source_text,
        }

    def describe(self) -> str:
        """One plain-ASCII line: owner, task and due date."""
        owner = self.owner or "unassigned"
        line = "%s: %s" % (owner, self.text)
        if self.due:
            line += " (due %s)" % self.due
        return line


@dataclass
class Question:
    """A question that was asked, and the later turn that answered it (if any)."""

    text: str
    speaker: Optional[str] = None
    turn_index: int = -1
    start: Optional[float] = None
    answered: bool = False
    answer_text: Optional[str] = None
    answered_by: Optional[str] = None
    answer_turn_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view of the question."""
        return {
            "text": self.text,
            "speaker": self.speaker,
            "turn_index": self.turn_index,
            "start": self.start,
            "answered": self.answered,
            "answer_text": self.answer_text,
            "answered_by": self.answered_by,
            "answer_turn_index": self.answer_turn_index,
        }


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

# (name, regex on the case-folded sentence, confidence). Documented in the
# README; read DECISION_CUES at runtime to see exactly what is matched.
DECISION_CUES = (
    ("we decided", r"\bwe(?:'ve| have| had)? ?decided\b", 0.92),
    ("decision is", r"\b(?:the |final |our )?decision (?:is|was)\b", 0.90),
    ("let us go with", r"\blet(?:'s| us| s)? go with\b", 0.88),
    ("agreed to", r"\bagreed (?:to|on|that)\b", 0.86),
    ("we agree", r"\bwe(?:'re| are|'ve| have)? ?(?:all )?agreed?\b", 0.82),
    ("the plan is", r"\b(?:the |our )?plan is\b", 0.80),
    ("we will go with", r"\bwe(?:'ll| will| are going to| shall) go with\b", 0.88),
    ("settled on", r"\bwe(?:'ve| have)? ?settled on\b", 0.86),
    ("we chose", r"\bwe (?:chose|picked|selected)\b", 0.84),
    ("going with", r"\bgoing with\b", 0.70),
    ("sign off", r"\bsign(?:ed)? off on\b", 0.78),
    ("approved", r"\b(?:is |was |are |were )approved\b", 0.76),
    ("consensus", r"\bconsensus (?:is|was)\b", 0.84),
    ("that is settled", r"\bthat(?:'s| is) (?:settled|decided|agreed)\b", 0.84),
    ("we will use", r"\bwe(?:'ll| will) (?:use|ship|adopt|keep|drop|move to)\b", 0.72),
    ("final call", r"\bfinal (?:call|answer|decision)\b", 0.80),
)

_DECISION_PATTERNS = tuple(
    (name, re.compile(pattern), weight) for name, pattern, weight in DECISION_CUES
)

# A decision cue inside any of these is not a decision.
_DECISION_BLOCKERS = re.compile(
    r"\b(?:have ?n[o']t|has ?n[o']t|had ?n[o']t|did ?n[o']t|do ?n[o']t|does ?n[o']t"
    r"|not yet|never|no)\s+(?:\w+\s+){0,2}(?:decided?|agreed?|settled|chosen)\b"
    r"|\bstill (?:deciding|discussing|debating|undecided|open)\b"
    r"|\bundecided\b|\bno decision\b|\bnot a decision\b"
    r"|\bonce we(?:'ve| have)? decided\b|\bbefore we decide\b"
)

# Hedges that lower confidence rather than block the match.
_HEDGES = re.compile(r"\b(?:maybe|perhaps|probably|might|could|possibly|i think|tentatively|for now|if we)\b")


def extract_decisions(turns: Sequence[Turn]) -> List[Decision]:
    """Find decision sentences using :data:`DECISION_CUES`.

    A cue inside a negation ("we have not decided", "still deciding") or inside
    a question is skipped, so an open item is never reported as settled.
    """
    decisions = []  # type: List[Decision]
    seen = set()
    for turn in turns:
        for sentence in turn.sentences():
            stripped = sentence.strip()
            if not stripped or stripped.endswith("?"):
                continue
            folded = fold(stripped)
            if _DECISION_BLOCKERS.search(folded):
                continue
            best = None  # type: Optional[Tuple[float, str]]
            for name, pattern, weight in _DECISION_PATTERNS:
                if pattern.search(folded):
                    if best is None or weight > best[0]:
                        best = (weight, name)
            if best is None:
                continue
            if word_count(stripped) < 3:
                continue
            # A decision restated by a second speaker is a confirmation, not a new
            # decision, so decisions stay keyed on the words alone. Actions do not:
            # see the comment on the action key below.
            key = folded
            if key in seen:
                continue
            seen.add(key)
            confidence = best[0]
            if _HEDGES.search(folded):
                confidence = max(0.35, confidence - 0.25)
            decisions.append(
                Decision(
                    text=stripped,
                    speaker=turn.speaker,
                    cue=best[1],
                    confidence=round(confidence, 3),
                    turn_index=turn.index,
                    start=turn.start,
                )
            )
    return decisions


# --------------------------------------------------------------------------
# Action items
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ActionRule:
    """One action cue: how it matches, who it implies, how much it is trusted."""

    name: str
    pattern: str
    weight: float
    owner: str = "none"        # speaker | addressee | capture | none
    cased: bool = False        # match against the original-case sentence
    allow_question: bool = False
    veto: str = ""             # a match here means this rule does not fire
    needs_owner: bool = False  # only meaningful when a real person is named


# "I'll ..." is a commitment only when a verb of doing follows it. These are the
# continuations that make it a remark instead: "I'll be honest", "I'll say the
# same", "I'll second that", "I'll admit", "I'll bet", "I'll leave it there".
# Without this veto the bare pronoun turns ordinary retro and standup speech
# into high-confidence tasks that nobody ever agreed to do.
_I_WILL_STANCE = (
    r"\bi(?:'ll| will| shall)\s+"
    r"(?:just |also |only |simply |probably |honestly |happily |gladly |quickly )*"
    r"(?:be|say|admit|confess|tell|bet|wager|guess|reckon|suppose|assume|second|echo"
    r"|agree|disagree|note|grant|concede|repeat|argue|venture|hazard|believe|hope"
    r"|see|think|know|wait|let|add that|point out|leave it|stop there|shut up)\b"
)

# Published so callers can read exactly which phrases create an action item.
ACTION_RULES = (
    _ActionRule("I'll take", r"\bi(?:'ll| will| can|'ve got| got)\s+(?:take|pick up|handle|own|grab|do|run|drive|cover)\b", 0.90, "speaker"),
    _ActionRule("let me", r"\blet me\s+(?!know\b)(?:\w+)\b", 0.74, "speaker"),
    _ActionRule("I will", r"\bi(?:'ll| will| shall)\s+\w+", 0.80, "speaker", veto=_I_WILL_STANCE),
    _ActionRule("I can", r"\bi can\s+(?:\w+)\b", 0.62, "speaker"),
    _ActionRule(
        "X will",
        r"\b(?P<who>(?!I\b)[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)?)\s+"
        r"(?:will|'ll|shall|is going to|is gonna)\s+(?!be able|have to\b)\w+",
        0.86,
        "capture",
        cased=True,
        needs_owner=True,
    ),
    _ActionRule("assigned to X", r"\bassigned to\s+(?P<who>[\w'’.-]+)", 0.88, "capture", cased=True),
    _ActionRule("X owns", r"\b(?P<who>[A-Z][\w'’.-]*)\s+(?:owns|takes|is on|picks up|has)\s+(?:this|that|it|the)\b", 0.82, "capture", cased=True),
    _ActionRule("can you", r"\b(?:can|could|would|will)\s+you\b", 0.74, "addressee", allow_question=True),
    _ActionRule("Name, can you", r"^\s*(?P<who>[^\W\d_][\w’.-]*),\s+(?:can|could|would|will|please)\b", 0.82, "capture", cased=True, allow_question=True, needs_owner=True),
    _ActionRule("please", r"\bplease\s+(?:\w+)\b", 0.66, "addressee"),
    _ActionRule("action item", r"\baction(?: item)?\b\s*[:\-]|\bthe action is\b|\btake an action\b", 0.86, "none"),
    _ActionRule("follow up", r"\bfollow[- ]?up\b|\bfollow up on\b", 0.64, "none"),
    _ActionRule("we need to", r"\bwe (?:need to|have to|must|should)\b", 0.55, "none"),
    _ActionRule("needs to be", r"\bneeds? to be\s+(?:\w+)\b", 0.58, "none"),
    _ActionRule("make sure", r"\b(?:make sure|don't forget to|remember to|be sure to)\b", 0.60, "none"),
    _ActionRule("todo", r"\bto-?do\b", 0.70, "none"),
)

_COMPILED_ACTION_RULES = tuple(
    (
        rule,
        re.compile(rule.pattern, re.UNICODE),
        re.compile(rule.veto, re.UNICODE) if rule.veto else None,
    )
    for rule in ACTION_RULES
)

# Capitalised words that are never an owner.
_NOT_A_NAME = frozenset("""
i we you he she they it this that there the a an and but so then now next
everyone everybody someone somebody anyone nobody all both each either neither
let lets please ok okay yes no yeah right well also just maybe perhaps
today tomorrow yesterday monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
if when while because since after before until unless however therefore
hi hey hello morning afternoon evening goodbye bye cheers thanks thank sorry
hmm hm mmm mm uh um er ah oh ooh oops wow ouch huh eh yep yup nope nah
guys folks team everybody alright actually honestly basically literally
wait hold hang look listen see sure fine great good cool nice perfect excellent
anyway anyhow besides meanwhile otherwise regardless nonetheless hopefully
first third finally lastly again still already almost nearly
""".split())

# "Someone will write it" names a person without naming which one, and that is
# still a commitment. "Hopefully Monday will be quieter" names nobody at all.
_INDEFINITE_PEOPLE = frozenset(
    "someone somebody anyone anybody everyone everybody nobody we they you".split()
)

# Published date cues. Each is tried in order; the first match wins.
DUE_PATTERNS = (
    r"\bby\s+(?:the\s+)?(?:end of (?:the )?(?:day|week|month|quarter|sprint))\b",
    r"\b(?:eod|eow|eom|cob)\b",
    r"\b(?:by|before|due|on|this|next)\s+(?:this |next |coming )?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:by|before|due)\s+(?:today|tomorrow|tonight|noon|midnight)\b",
    r"\b(?:by|before|due|on)\s+\d{4}-\d{2}-\d{2}\b",
    r"\b(?:by|before|due|on)\s+\d{1,2}[/.]\d{1,2}(?:[/.]\d{2,4})?\b",
    r"\b(?:by|before|due|on)\s+(?:the\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
    r"\b(?:by|before|due|on)\s+\d{1,2}(?:st|nd|rd|th)\s+(?:of\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(?:by|before|due|on)\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\b",
    r"\b(?:by|before)\s+\d{1,2}\s*(?:am|pm)\b",
    r"\b(?:within|in)\s+(?:the\s+)?(?:next\s+)?(?:\d+|a|two|three|four|five|six|seven|ten)\s+(?:minutes?|hours?|days?|weeks?|months?)\b",
    r"\b(?:next|this)\s+(?:week|month|quarter|sprint|monday|tuesday|wednesday|thursday|friday)\b",
    r"\bend of (?:the )?(?:day|week|month|quarter|sprint)\b",
    r"\b(?:asap|immediately|right away|straight away)\b",
)

_COMPILED_DUE = tuple(re.compile(pattern) for pattern in DUE_PATTERNS)

_DUE_LEADERS = re.compile(r"^(?:by|before|due|on|in)\s+")

# Restored to their usual written form, because the match ran on folded text.
_DUE_CAPITALS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
""".split())
_DUE_UPPER = frozenset(("eod", "eow", "eom", "cob", "asap"))


def _present_due(phrase: str) -> str:
    """Write a matched due phrase the way a person would: Friday, EOD, ASAP."""
    words = []
    for word in phrase.split():
        core = word.strip(".,")
        if core in _DUE_UPPER:
            words.append(word.replace(core, core.upper()))
        elif core in _DUE_CAPITALS:
            words.append(word.replace(core, core.capitalize()))
        else:
            words.append(word)
    return " ".join(words)

_ACTION_BLOCKERS = re.compile(
    r"\bi(?:'ll| will) (?:be|have|get) (?:back|going|there)\b"
    r"|\bi(?:'ll| will) (?:see|check|think about|let you know)\b"
    r"|\bthanks\b|\bthank you\b"
    r"|\b(?:i|we) (?:already|just) (?:did|sent|finished|shipped)\b"
    # Call logistics. "Hey, can you hear me?" is the most ordinary line in any
    # remote meeting and it is not a task anybody has to do afterwards.
    r"|\bcan (?:you|anyone|everyone|somebody|someone) (?:still )?(?:hear|see)\b"
    r"|\b(?:you|we)(?:'re| are) (?:on mute|breaking up|frozen|cutting out)\b"
    r"|\bam i (?:audible|coming through|on mute)\b"
)


def find_due(text: str) -> Optional[str]:
    """Return the due-date phrase in ``text``, or ``None``.

    Matched against :data:`DUE_PATTERNS`; the leading "by"/"before"/"due" is
    trimmed so the value reads as a date ("Friday", "end of week", "2026-04-01").
    """
    folded = fold(text)
    for pattern in _COMPILED_DUE:
        match = pattern.search(folded)
        if match:
            phrase = _DUE_LEADERS.sub("", match.group(0).strip())
            phrase = phrase.replace("the ", "", 1) if phrase.startswith("the ") else phrase
            phrase = phrase.strip()
            return _present_due(phrase) if phrase else None
    return None


def _match_speaker(candidate: str, speakers: Sequence[str]) -> Optional[str]:
    """Resolve a bare name against the speaker list, tolerating first names."""
    if not candidate:
        return None
    needle = candidate.strip().strip(".,:;'’").lower()
    if not needle:
        return None
    for speaker in speakers:
        if speaker.lower() == needle:
            return speaker
    for speaker in speakers:
        parts = speaker.lower().split()
        if needle in parts:
            return speaker
        if parts and (parts[0].startswith(needle) or needle.startswith(parts[0])):
            if abs(len(parts[0]) - len(needle)) <= 3:
                return speaker
    return None


# A name written next to the request: "Alice, can you ..." and the identical
# trailing form "Can you ..., Alice?". Both name the addressee, so both have to
# be read before falling back to whoever happens to speak next.
_VOCATIVE_LEAD = re.compile(r"^\s*(?P<who>[^\W\d_][\w'’.-]*)\s*,", re.UNICODE)
_VOCATIVE_TAIL = re.compile(r",\s*(?P<who>[^\W\d_][\w'’.-]*)\s*[?.!]*\s*$", re.UNICODE)


def _vocative(sentence: str, speakers: Sequence[str]) -> Optional[str]:
    """The roster name this sentence addresses, leading or trailing, else ``None``."""
    if not speakers:
        return None
    for pattern in (_VOCATIVE_LEAD, _VOCATIVE_TAIL):
        match = pattern.search(sentence)
        if match is None:
            continue
        owner, known = _capture_owner(match.group("who"), speakers)
        if known:
            return owner
    return None


def _addressee(
    turn: Turn,
    turns: Sequence[Turn],
    speakers: Sequence[str],
    sentence: str = "",
) -> Optional[str]:
    """Who "can you ..." is aimed at.

    A name written into the sentence wins, whether it leads ("Alice, can you
    review it?") or trails ("Can you review it, Alice?"). The two forms mean
    the same thing and must not land on different people. Failing that, with
    exactly two people in the room it is the other one, and otherwise it is the
    next person to speak, which is who actually answers in practice.
    """
    named = _vocative(sentence, speakers) if sentence else None
    if named:
        return named
    known = [name for name in speakers if name]
    if turn.speaker and len(known) == 2:
        for name in known:
            if name != turn.speaker:
                return name
    for following in turns[turn.index + 1: turn.index + 4]:
        if following.speaker and following.speaker != turn.speaker:
            return following.speaker
    return None


def _capture_owner(raw: str, speakers: Sequence[str]) -> Tuple[Optional[str], bool]:
    """Resolve a captured name; returns ``(owner, matched_a_known_speaker)``.

    A capitalised word is not evidence that a person exists. "Hey", "Sorry",
    "Team", "Folks" and "Hopefully Monday" are all capitalised and none of them
    is an owner, so a capture becomes an owner only when it matches the roster:
    the speakers found in the transcript plus any names handed to
    ``analyse(speakers=...)``. Anything else is left ``None`` rather than
    guessed, which is also what happens all the way through a transcript that
    carries no speaker labels and so has no roster to check against.
    """
    candidate = raw.strip().strip(".,:;'’")
    if not candidate:
        return None, False
    lowered = candidate.lower()
    for speaker in speakers:
        if speaker and speaker.lower() == lowered:
            return speaker, True
    parts = [part.lower() for part in candidate.split()]
    if lowered in _NOT_A_NAME or any(part in _NOT_A_NAME for part in parts):
        return None, False
    matched = _match_speaker(candidate, speakers)
    if matched:
        return matched, True
    return None, False


def _clean_action_text(sentence: str) -> str:
    """Tidy a sentence into a task line without changing its meaning."""
    text = " ".join(sentence.split())
    text = re.sub(r"^(?:and|so|ok|okay|right|well|um|uh|yeah|yes|no)[,\s]+", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_actions(
    turns: Sequence[Turn],
    speakers: Sequence[str] = (),
) -> List[Action]:
    """Find action items using :data:`ACTION_RULES`.

    Owners are resolved against ``speakers``: "I" and "I'll" become the speaker
    of the turn, and "can you" becomes the person addressed -- a name written
    into the sentence when there is one ("Alice, can you ..." and "Can you ...,
    Alice?" both name Alice), otherwise the other person on a two-person call or
    the next person to speak. A named owner has to match somebody in
    ``speakers``; a capitalised word that matches nobody is not treated as a
    person, so an owner that cannot be resolved is left ``None`` rather than
    guessed. That is also what happens throughout a transcript that carries no
    speaker labels at all and so has nothing to resolve against.

    A rule that only means something with a person attached -- "X will", and a
    leading vocative -- stands down when the name is not a known person, and the
    next-best rule is tried instead. "Someone will ..." is the exception: it
    names a person without saying which, so it keeps the action and leaves the
    owner as ``None``.
    """
    actions = []  # type: List[Action]
    seen = set()
    for turn in turns:
        for sentence in turn.sentences():
            stripped = sentence.strip()
            if not stripped or word_count(stripped) < 3:
                continue
            folded = fold(stripped)
            if _ACTION_BLOCKERS.search(folded):
                continue
            is_question = stripped.endswith("?")
            candidates = []  # type: List[Tuple[float, int, _ActionRule, Any]]
            for order, (rule, pattern, veto) in enumerate(_COMPILED_ACTION_RULES):
                if is_question and not rule.allow_question:
                    continue
                match = pattern.search(stripped if rule.cased else folded)
                if match is None:
                    continue
                if veto is not None and veto.search(folded):
                    continue
                candidates.append((rule.weight, order, rule, match))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (-item[0], item[1]))

            chosen = None  # type: Optional[Tuple[_ActionRule, Optional[str], bool, bool]]
            for _weight, _order, rule, match in candidates:
                owner = None  # type: Optional[str]
                owner_known = False
                owner_is_speaker = False
                captured = ""
                if rule.owner == "speaker":
                    owner = turn.speaker
                    owner_is_speaker = owner is not None
                    owner_known = owner is not None
                elif rule.owner == "addressee":
                    owner = _addressee(turn, turns, speakers, stripped)
                    owner_known = owner is not None
                elif rule.owner == "capture":
                    try:
                        captured = match.group("who") or ""
                    except IndexError:  # pragma: no cover - capture rules name the group
                        captured = ""
                    owner, owner_known = _capture_owner(captured, speakers)
                # A rule whose whole meaning is "this named person will do it"
                # says nothing once the name turns out not to be a person. Let
                # the next-best rule speak instead of inventing an owner. An
                # indefinite person ("someone will ...") is still a person, so
                # it keeps the action and leaves the owner as None.
                if rule.needs_owner and speakers and not owner_known:
                    if captured.strip().lower() not in _INDEFINITE_PEOPLE:
                        continue
                chosen = (rule, owner, owner_known, owner_is_speaker)
                break
            if chosen is None:
                continue
            rule, owner, owner_known, owner_is_speaker = chosen

            due = find_due(stripped)
            confidence = rule.weight
            if due:
                confidence += 0.06
            if owner:
                confidence += 0.04
            if owner_known:
                confidence += 0.03
            if _HEDGES.search(folded):
                confidence -= 0.22
            confidence = round(max(0.2, min(0.97, confidence)), 3)

            text = _clean_action_text(stripped)
            # Key on the SPEAKER as well as the words. Two people committing to the
            # same thing is two commitments, and keying on the text alone silently
            # dropped the second one - the very case a meeting tool must not lose,
            # because the dropped person is the one who never gets chased.
            key = (owner or "", fold(text))
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                Action(
                    text=text,
                    owner=owner,
                    due=due,
                    confidence=confidence,
                    speaker=turn.speaker,
                    cue=rule.name,
                    turn_index=turn.index,
                    start=turn.start,
                    owner_is_speaker=owner_is_speaker,
                )
            )
    return actions


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------

_AFFIRMATIONS = re.compile(
    r"^(?:yes|yeah|yep|yup|no|nope|sure|right|correct|exactly|agreed|true|false"
    r"|absolutely|definitely|of course|i think|i believe|it(?:'s| is)|we (?:can|could|did|do|will|have)"
    r"|that(?:'s| is)|there (?:is|are|was|were)|probably|about|around|roughly)\b"
)

# A bare affirmation only answers a question when it comes straight back.
# Six turns later, "That is everything." matches on "that is" and is sign-off
# language, not an answer to anything.
_ANSWER_WINDOW_TURNS = 6
_DIRECT_ANSWER_WINDOW_TURNS = 2

# Winding the meeting up. Never an answer, whatever it happens to start with.
_CLOSING_REMARKS = re.compile(
    r"\bthat(?:'s| is) (?:everything|all|it|us|the lot)\b"
    r"|\bthanks (?:all|everyone|both|guys|folks)\b"
    r"|\bwe(?:'re| are) (?:done|out of time|over time)\b"
    r"|\blet(?:'s| us) (?:wrap|call it|leave it there)\b"
    r"|\b(?:nothing|anything) else from (?:me|anyone)\b"
    r"|\bany other business\b|\bsee you (?:next|tomorrow|later)\b"
)

# "Who owns that?" is answered by naming somebody, not by agreeing.
_WHO_QUESTION = re.compile(r"^(?:so |and |ok |okay |right )*(?:who|whose|who's)\b")
_ASSIGNMENT_CUE = re.compile(
    r"\bassigned to\b|\b(?:owns|owned by|takes|took|has|picked up|picking up) (?:it|that|this)\b"
    r"|\bthat(?:'s| is) (?:mine|me|yours|ours|on me)\b|\bi(?:'ll| will| am| can) (?:own|take|do) (?:it|that|this)\b"
)


def _is_real_question(sentence: str) -> bool:
    """Filter out tag questions and filler ("Right?", "Okay?")."""
    if not sentence.endswith("?"):
        return False
    tokens = tokenize(sentence)
    if len(tokens) < 3:
        return False
    if all(token in STOP_WORDS for token in tokens):
        return False
    return True


def extract_questions(turns: Sequence[Turn]) -> List[Question]:
    """Find questions and check whether a later turn answers each one.

    A question is a sentence ending in "?" with at least three words, at least
    one of which carries meaning -- that drops "Right?" and "Okay?". A later
    turn counts as the answer when a different person replies within six turns
    and shares a content word with the question, when a "who" question is
    answered by naming somebody, or when the very next turns open with a direct
    answer ("yes", "no", "it is", "we can"). A bare affirmation further away,
    and any closing remark, is not treated as an answer: leaving the question
    open is the honest result, and the summary counts it as open.
    """
    names = []  # type: List[str]
    for turn in turns:
        if turn.speaker and turn.speaker not in names:
            names.append(turn.speaker)
    questions = []  # type: List[Question]
    for turn in turns:
        for sentence in turn.sentences():
            stripped = sentence.strip()
            if not _is_real_question(stripped):
                continue
            question = Question(
                text=stripped,
                speaker=turn.speaker,
                turn_index=turn.index,
                start=turn.start,
            )
            _match_answer(question, turn, turns, names)
            questions.append(question)
    return questions


def _names_somebody(folded: str, names: Sequence[str]) -> bool:
    """True when the reply names a person or says who the work went to."""
    if _ASSIGNMENT_CUE.search(folded):
        return True
    for name in names:
        for part in fold(name).split():
            if len(part) > 1 and re.search(r"\b%s\b" % re.escape(part), folded):
                return True
    return False


def _match_answer(
    question: Question,
    asked_in: Turn,
    turns: Sequence[Turn],
    names: Sequence[str] = (),
) -> None:
    """Fill in the answer fields on ``question`` when a later turn answers it."""
    asked_words = set(content_words(question.text))
    asks_who = bool(_WHO_QUESTION.match(fold(question.text)))
    window = turns[asked_in.index + 1: asked_in.index + 1 + _ANSWER_WINDOW_TURNS]
    for candidate in window:
        if asked_in.speaker is not None and candidate.speaker == asked_in.speaker:
            continue
        body = candidate.text.strip()
        if not body:
            continue
        folded = fold(body)
        sentences = candidate.sentences()
        non_questions = [item for item in sentences if not item.strip().endswith("?")]
        if not non_questions:
            continue
        distance = candidate.index - asked_in.index
        closing = bool(_CLOSING_REMARKS.search(folded))
        shared = asked_words & set(content_words(body))
        named = asks_who and _names_somebody(folded, names)
        direct = (
            bool(_AFFIRMATIONS.match(folded))
            and distance <= _DIRECT_ANSWER_WINDOW_TURNS
            and not closing
        )
        if not shared and not named and not direct:
            continue
        if word_count(body) < 2:
            continue
        question.answered = True
        question.answer_text = non_questions[0].strip()
        question.answered_by = candidate.speaker
        question.answer_turn_index = candidate.index
        return


def unanswered(questions: Sequence[Question]) -> List[Question]:
    """The subset of ``questions`` that nobody answered."""
    return [question for question in questions if not question.answered]
