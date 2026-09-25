"""The fuzzy matcher: align what was said with a pattern and score the fit.

How it works (a heuristic, not a language model):

1. The utterance is tokenised; hesitations ("um", "uh") are dropped and filler
   phrases ("could you", "please", "just") are marked as free to skip.
2. A dynamic-programming alignment walks the pattern and the words together.
   Each pattern element may match a word (fuzzily, by edit distance, a plural,
   a doubled letter or a sound-alike), a split or joined word, a number span,
   one of its choices, or a run of free text, or be left missing. Each word may
   be consumed or left over; left-over content words cost evidence, left-over
   fillers cost nothing.
3. A reorder pass lets a missing fixed word, number or choice claim a left-over
   word said somewhere else ("turn the lights off" for "turn {state:on|off} the
   {device}"), at a small cost.
4. Confidence = coverage^1.5 * precision^0.5, times 0.85 when no fixed word was
   heard exactly. Coverage is the weighted share of the pattern that was found
   (a fuzzy word earns its similarity squared, and a word contradicted by a
   different word in its place loses extra); precision is the weighted share of
   the words that the pattern explains, with chatter before or after the
   command counting half.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._numbers import number_spans, parse_keys
from ._pattern import Element, Pattern
from ._text import FUNCTION_WORDS, Token, similarity, tokenize

LEFTOVER_COST = 0.35  # alignment cost of leaving one content word unexplained
REORDER_CREDIT = 0.85  # credit kept by an element found out of order
NO_ANCHOR_FACTOR = 0.85  # applied when no fixed word was heard exactly
COVERAGE_POWER = 1.5
PRECISION_POWER = 0.5
OUTSIDE_WEIGHT = 0.5  # chatter before/after the command counts half
OPTIONAL_CREDIT = 0.2  # alignment reward for consuming an optional word
JOIN_MIN = 0.8  # similarity needed to accept a split or joined word
JOIN_FACTOR = 0.95
TEXT_BONUS = 1e-4  # makes text slots greedy when everything else is equal
MAX_NUMBER_WORDS = 12  # longest spoken number considered ("one thousand two hundred ...")
REPLACED_PENALTY = 0.5  # extra loss when another word was said in a missing word's place

_WS_RE = re.compile(r"\s+")


@dataclass
class Utterance:
    """What was said, prepared once and matched against every pattern."""

    original: str
    text: str
    tokens: List[Token]
    dropped: List[Token]
    numbers: List[List[Tuple[int, float, bool]]]

    @property
    def keys(self) -> List[str]:
        return [t.key for t in self.tokens]


def prepare(text: str, fillers: Dict[Tuple[str, ...], str]) -> Utterance:
    """Normalise and tokenise ``text`` for matching."""
    nfc = unicodedata.normalize("NFC", text)
    tokens, dropped = tokenize(nfc, fillers)
    numbers = number_spans([t.key for t in tokens], MAX_NUMBER_WORDS)
    return Utterance(text, nfc, tokens, dropped, numbers)


@dataclass
class ElementResult:
    """How one pattern element was found (or not) in the utterance."""

    element: Element
    status: str = "missing"  # "matched", "reordered", "missing" or "absent"
    start: int = -1
    end: int = -1  # token range [start, end)
    credit: float = 0.0
    score: float = 0.0  # similarity of the words to what was expected
    how: str = ""
    value: Any = None

    @property
    def found(self) -> bool:
        return self.status in ("matched", "reordered")


@dataclass
class Alignment:
    """The best alignment of one pattern with one utterance, and its scores."""

    pattern: Pattern
    utterance: Utterance
    results: List[ElementResult]
    leftovers: List[int]  # token indices not explained by the pattern
    trimmed: List[int] = field(default_factory=list)  # fillers trimmed off text slots
    #: missing elements with other words said in their place: (result, token indices)
    replaced: List[Tuple[ElementResult, List[int]]] = field(default_factory=list)
    coverage: float = 0.0
    precision: float = 0.0
    confidence: float = 0.0
    anchored: bool = True


def credit(sim: float) -> float:
    """Credit for a word heard with similarity ``sim``: squared, so typos cost more
    than their raw similarity suggests and a lone fuzzy word proves little."""
    return sim * sim


def _substantive(tokens: Sequence[Token], start: int, end: int) -> bool:
    """True when tokens [start, end) hold something other than politeness or framing."""
    return any(tokens[k].kind in ("", "soft") for k in range(start, end))


def _span_text(utt: Utterance, start: int, end: int) -> str:
    """Original text of tokens [start, end), without dropped hesitations."""
    toks = utt.tokens[start:end]
    if not toks:
        return ""
    pieces = [toks[0].surface]
    for prev, cur in zip(toks, toks[1:]):
        gap = utt.text[prev.end:cur.start]
        if any(prev.end <= d.start and d.end <= cur.start for d in utt.dropped):
            gap = " "
        pieces.append(_WS_RE.sub(" ", gap))
        pieces.append(cur.surface)
    return "".join(pieces)


def _dp(pattern: Pattern, utt: Utterance) -> List[tuple]:
    """Run the alignment and return its steps in order."""
    elements = pattern.elements
    tokens = utt.tokens
    m, n = len(elements), len(tokens)
    neg = float("-inf")
    score = [[neg] * (n + 1) for _ in range(m + 1)]
    back: List[List[Optional[tuple]]] = [[None] * (n + 1) for _ in range(m + 1)]
    score[0][0] = 0.0

    def relax(j: int, i: int, value: float, step: tuple) -> None:
        if value > score[j][i] + 1e-12:
            score[j][i] = value
            back[j][i] = step

    for j in range(m + 1):
        element = elements[j] if j < m else None
        for i in range(n + 1):
            s = score[j][i]
            if s == neg:
                continue
            if i < n:
                relax(j, i + 1, s - LEFTOVER_COST * tokens[i].weight, ("leftover", j, i, None))
            if element is None:
                continue
            relax(j + 1, i, s, ("skip", j, i, None))
            if i == n:
                continue
            kind = element.kind
            if kind in ("literal", "optional"):
                w = element.weight if kind == "literal" else OPTIONAL_CREDIT
                sim, how = similarity(element.key, tokens[i].key)
                if sim > 0:
                    relax(j + 1, i + 1, s + w * credit(sim), ("word", j, i, (1, 1, sim, how)))
                if i + 1 < n:
                    sim, _ = similarity(element.key, tokens[i].key + tokens[i + 1].key)
                    if sim >= JOIN_MIN:
                        sim *= JOIN_FACTOR
                        relax(j + 1, i + 2, s + w * credit(sim),
                              ("word", j, i, (1, 2, sim, "split into two words")))
                if j + 1 < m and elements[j + 1].kind in ("literal", "optional"):
                    nxt = elements[j + 1]
                    w2 = nxt.weight if nxt.kind == "literal" else OPTIONAL_CREDIT
                    sim, _ = similarity(element.key + nxt.key, tokens[i].key)
                    if sim >= JOIN_MIN:
                        sim *= JOIN_FACTOR
                        relax(j + 2, i + 1, s + (w + w2) * credit(sim),
                              ("word", j, i, (2, 1, sim, "run together as one word")))
            elif kind == "number":
                for k, value, whole in utt.numbers[i]:
                    relax(j + 1, i + k, s + element.weight, ("number", j, i, (k, value, whole)))
            elif kind == "choice":
                for index, (_, keys) in enumerate(element.options):
                    q = len(keys)
                    if i + q > n:
                        continue
                    sims = []
                    for r in range(q):
                        sim, how = similarity(keys[r], tokens[i + r].key)
                        if sim <= 0:
                            break
                        sims.append((sim, how))
                    else:
                        mean = sum(x for x, _ in sims) / q
                        worst = min(sims)[1]
                        relax(j + 1, i + q, s + element.weight * credit(mean),
                              ("choice", j, i, (q, index, mean, worst)))
            elif kind == "word":
                tok = tokens[i]
                factor = 1.0 if tok.kind in ("", "soft") and tok.key not in FUNCTION_WORDS else 0.5
                relax(j + 1, i + 1, s + element.weight * factor, ("slot", j, i, (1, factor)))
            else:  # text: any run holding more than politeness and framing
                substantive = False
                for k in range(1, n - i + 1):
                    substantive = substantive or tokens[i + k - 1].kind in ("", "soft")
                    if substantive:
                        relax(j + 1, i + k, s + element.weight + TEXT_BONUS * k,
                              ("slot", j, i, (k, 1.0)))
    steps: List[tuple] = []
    j, i = m, n
    while (j, i) != (0, 0):
        step = back[j][i]
        if step is None:  # pragma: no cover - (0, 0) always reaches (m, n)
            break
        steps.append(step)
        j, i = step[1], step[2]
    steps.reverse()
    return steps


def _number_value(value: float, whole: bool) -> Any:
    return int(value) if whole and float(value).is_integer() else float(value)


def _recover(alignment: Alignment) -> None:
    """Let missing fixed words, numbers and choices claim words said out of order."""
    utt = alignment.utterance
    tokens = utt.tokens
    keys = utt.keys
    leftovers = set(alignment.leftovers)
    for res in alignment.results:
        element = res.element
        if res.status != "missing" or element.kind not in ("literal", "number", "choice"):
            continue
        # best = (rank, start, end, value, score, how, text slot it came from or None);
        # rank prefers more credit, then left-over words over text-slot edges, then earlier
        best: Optional[tuple] = None

        def offer(earned: float, start: int, end: int, value: Any, score: float,
                  how: str, source: Optional[ElementResult]) -> None:
            nonlocal best
            rank = (earned, source is None, -start)
            if best is None or rank > best[0]:
                best = (rank, start, end, value, score, how, source)

        runs: List[List[int]] = []  # maximal runs [start, end) of left-over words
        for i in sorted(leftovers):
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        text_slots = [r for r in alignment.results if r.found and r.element.kind == "text"]

        if element.kind == "literal":
            for start, end in runs:
                for i in range(start, end):
                    if tokens[i].is_filler:
                        continue
                    sim, how = similarity(element.key, tokens[i].key)
                    if sim > 0:
                        offer(element.weight * credit(sim), i, i + 1, None, sim, how, None)
        else:
            if element.kind == "number":
                longest = MAX_NUMBER_WORDS
            else:
                longest = max(len(keys_) for _, keys_ in element.options)
            windows: List[Tuple[int, int, Optional[ElementResult]]] = []
            for start, end in runs:
                for a in range(start, end):
                    for b in range(a + 1, min(end, a + longest) + 1):
                        windows.append((a, b, None))
            for slot in text_slots:
                for q in range(1, min(slot.end - slot.start, longest + 1)):
                    if _substantive(tokens, slot.start, slot.end - q):
                        windows.append((slot.end - q, slot.end, slot))
                    if _substantive(tokens, slot.start + q, slot.end):
                        windows.append((slot.start, slot.start + q, slot))
            for a, b, source in windows:
                if element.kind == "number":
                    parsed = parse_keys(keys[a:b])
                    if parsed is not None:
                        # longer spans first: "twenty one" beats "twenty"
                        offer(element.weight + 1e-3 * (b - a), a, b,
                              _number_value(*parsed), 1.0, "exact", source)
                else:
                    for canon, opt_keys in element.options:
                        if len(opt_keys) != b - a:
                            continue
                        sims = [similarity(k, keys[a + r]) for r, k in enumerate(opt_keys)]
                        if all(s > 0 for s, _ in sims):
                            mean = sum(s for s, _ in sims) / len(sims)
                            offer(element.weight * credit(mean), a, b, canon, mean,
                                  min(sims)[1], source)
        if best is None:
            continue
        _, start, end, value, score, how, source = best
        res.status = "reordered"
        res.start, res.end = start, end
        res.credit = min(element.weight, best[0][0]) * REORDER_CREDIT
        res.score, res.how = score, how
        res.value = value if element.kind != "literal" else None
        if source is None:
            leftovers.difference_update(range(start, end))
        else:
            if start == source.start:
                source.start = end
            else:
                source.end = start
            source.value = _span_text(utt, source.start, source.end)
    alignment.leftovers = sorted(leftovers)


def _trim_text_slots(alignment: Alignment) -> None:
    """Drop politeness from the edges of text slots, and request framing from
    the start of a text slot that opens the pattern."""
    tokens = alignment.utterance.tokens
    first_required = next((r for r in alignment.results if r.element.required), None)
    for res in alignment.results:
        if res.element.kind != "text" or not res.found:
            continue
        start, end = res.start, res.end
        while end - start > 1 and tokens[end - 1].kind == "polite" and _substantive(tokens, start, end - 1):
            end -= 1
        if res is first_required:
            while (end - start > 1 and tokens[start].kind in ("polite", "frame")
                   and _substantive(tokens, start + 1, end)):
                start += 1
        alignment.trimmed.extend(range(res.start, start))
        alignment.trimmed.extend(range(end, res.end))
        res.start, res.end = start, end
        res.value = _span_text(alignment.utterance, start, end)


def _replaced(alignment: Alignment) -> List[Tuple[ElementResult, List[int]]]:
    """Missing fixed words, numbers and choices with another content word said in
    their place, between the elements around them ("turn OFF the light" for
    "turn on the light"). Being contradicted is worse than being left out."""
    tokens = alignment.utterance.tokens
    leftovers = [i for i in alignment.leftovers if tokens[i].weight >= 1.0]
    if not leftovers:
        return []
    results = alignment.results
    out: List[Tuple[ElementResult, List[int]]] = []
    for index, res in enumerate(results):
        if res.status != "missing" or res.element.kind not in ("literal", "number", "choice"):
            continue
        if res.element.weight < 1.0:
            continue
        before = [r for r in results[:index] if r.status == "matched"]
        after = [r for r in results[index + 1:] if r.status == "matched"]
        if not before or not after:
            continue
        lo, hi = before[-1].end, after[0].start
        instead = [i for i in leftovers if lo <= i < hi]
        if instead:
            out.append((res, instead))
    return out


def _score(alignment: Alignment) -> None:
    tokens = alignment.utterance.tokens
    required = [r for r in alignment.results if r.element.required]
    total = sum(r.element.weight for r in required)
    alignment.replaced = _replaced(alignment)
    earned = sum(r.credit for r in required) - REPLACED_PENALTY * sum(
        r.element.weight for r, _ in alignment.replaced
    )
    coverage = max(0.0, min(1.0, earned / total)) if total > 0 else 0.0

    used = set(alignment.trimmed)
    for r in alignment.results:
        if r.found:
            used.update(range(r.start, r.end))
    explained = sum(tokens[i].weight for i in used)
    leftover_weight = 0.0
    if used:
        lo, hi = min(used), max(used)
        for i in alignment.leftovers:
            factor = 1.0 if lo < i < hi else OUTSIDE_WEIGHT
            leftover_weight += tokens[i].weight * factor
    if not used:
        precision = 0.0
    elif explained + leftover_weight > 0:
        precision = explained / (explained + leftover_weight)
    else:
        precision = 1.0

    fixed = [r for r in required if r.element.kind == "literal" and r.element.weight >= 1.0]
    anchored = not fixed or any(r.found and r.score >= 0.95 for r in fixed)
    confidence = (coverage ** COVERAGE_POWER) * (precision ** PRECISION_POWER)
    if not anchored:
        confidence *= NO_ANCHOR_FACTOR
    alignment.coverage = coverage
    alignment.precision = max(0.0, min(1.0, precision))
    alignment.confidence = max(0.0, min(1.0, confidence))
    alignment.anchored = anchored


def align(pattern: Pattern, utt: Utterance) -> Alignment:
    """Best alignment of ``pattern`` with ``utt``, fully scored."""
    elements = pattern.elements
    results = [ElementResult(e) for e in elements]
    leftovers: List[int] = []
    for action, j, i, info in _dp(pattern, utt):
        if action == "leftover":
            leftovers.append(i)
        elif action == "skip":
            results[j].status = "missing" if elements[j].required else "absent"
        elif action == "word":
            n_elements, n_tokens, sim, how = info
            for r in range(n_elements):
                res = results[j + r]
                res.status = "matched"
                res.start, res.end = i, i + n_tokens
                weight = res.element.weight if res.element.required else 0.0
                res.credit = weight * credit(sim)
                res.score, res.how = sim, how
        elif action == "number":
            k, value, whole = info
            res = results[j]
            res.status, res.start, res.end = "matched", i, i + k
            res.credit, res.score, res.how = elements[j].weight, 1.0, "exact"
            res.value = _number_value(value, whole)
        elif action == "choice":
            q, index, mean, how = info
            res = results[j]
            res.status, res.start, res.end = "matched", i, i + q
            res.credit, res.score = elements[j].weight * credit(mean), mean
            res.how = "exact" if mean >= 1.0 else how
            res.value = elements[j].options[index][0]
        else:  # word or text slot
            k, factor = info
            res = results[j]
            res.status, res.start, res.end = "matched", i, i + k
            res.credit, res.score, res.how = elements[j].weight * factor, factor, "exact"
            if elements[j].kind == "word":
                res.value = utt.tokens[i].surface
            else:
                res.value = _span_text(utt, i, i + k)
    alignment = Alignment(pattern, utt, results, leftovers)
    _trim_text_slots(alignment)
    _recover(alignment)
    _score(alignment)
    return alignment


def describe_words(utt: Utterance, start: int, end: int) -> str:
    """The words of tokens [start, end) as heard."""
    return " ".join(t.surface for t in utt.tokens[start:end])


def ignored_phrases(alignment: Alignment) -> List[str]:
    """Filler words that were skipped, grouped into phrases, in spoken order."""
    tokens = alignment.utterance.tokens
    idle = sorted(
        {i for i in alignment.leftovers if tokens[i].is_filler} | set(alignment.trimmed)
    )
    groups: List[Tuple[int, str]] = []
    run: List[int] = []
    for i in idle:
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                groups.append((tokens[run[0]].start, describe_words(alignment.utterance, run[0], run[-1] + 1)))
            run = [i]
    if run:
        groups.append((tokens[run[0]].start, describe_words(alignment.utterance, run[0], run[-1] + 1)))
    groups.extend((d.start, d.surface) for d in alignment.utterance.dropped)
    return [text for _, text in sorted(groups)]
