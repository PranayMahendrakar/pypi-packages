"""How hard is this prompt? A small, documented, deterministic score.

``complexity(prompt)`` returns a :class:`Complexity` with a score between 0 and
1 and a band of ``"simple"``, ``"moderate"`` or ``"hard"``. Five signals feed
the score, each measured on its own 0-1 scale and then weighted:

============ ====== ==========================================================
signal       weight what it measures
============ ====== ==========================================================
length       0.15   words, reaching 1.0 at 60 words
questions    0.15   question marks plus question words, reaching 1.0 at 3
reasoning    0.35   reasoning words such as "explain" or "compare", 1.0 at 3
code         0.15   code markers such as a fenced block or "def ", 1.0 at 2
instructions 0.20   distinct instructions found, 1.0 at 4
============ ====== ==========================================================

Reasoning carries the most weight on purpose: a short prompt that asks for a
derivation is harder than a long one that pastes a log. Length is the smallest
term for the same reason.

A score under 0.22 is "simple", under 0.45 "moderate", and 0.45 or above
"hard". Nothing here is learned, random or downloaded: the same text always
gives the same score, on every machine, offline.

Scripts
-------

Words are counted in every script, not only Latin: text is split on Unicode
word characters, and a run of Chinese, Japanese or Korean - which is written
without spaces - counts one word per ``CJK_CHARS_PER_WORD`` characters rather
than collapsing to a single word.

The question, reasoning and chain markers are word lists, so each one only
fires in a language it is written for. They cover English plus Chinese,
Japanese, Korean, Russian, Arabic, Hindi, Greek and Hebrew. A prompt written
mostly in some other non-Latin script is still scored on its length and
structure, and the result then carries a plain-language entry in
:attr:`Complexity.warnings` saying the language markers could not be read. The
scorer never silently returns a degenerate 0.0 for text it could not read: it
says so, in the result and in the log.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Score below which a prompt is "simple".
SIMPLE_MAX = 0.22
#: Score below which a prompt is "moderate"; at or above it a prompt is "hard".
MODERATE_MAX = 0.45
#: Rough characters-per-token used for every token estimate in this package.
CHARS_PER_TOKEN = 4
#: Characters of a script written without spaces (Chinese, Japanese, Korean)
#: that count as one word.
CJK_CHARS_PER_WORD = 2
#: Share of a prompt's letters that must be non-Latin before it is treated as
#: written in another script.
NON_LATIN_SHARE = 0.5
#: Words a non-Latin prompt needs before "no marker matched" is worth warning
#: about. A three-word greeting really is simple, whatever the script.
WORDS_FOR_SCRIPT_WARNING = 8

#: Weight of each signal in the final score. The weights sum to 1.0.
SIGNAL_WEIGHTS = {
    "length": 0.15,
    "questions": 0.15,
    "reasoning": 0.35,
    "code": 0.15,
    "instructions": 0.20,
}

#: Word count at which the length signal reaches 1.0.
WORDS_FOR_FULL_LENGTH = 60
#: Question hits at which the question signal reaches 1.0.
QUESTIONS_FOR_FULL = 3
#: Distinct reasoning markers at which the reasoning signal reaches 1.0.
REASONING_FOR_FULL = 3
#: Distinct code markers at which the code signal reaches 1.0.
CODE_MARKERS_FOR_FULL = 2
#: Distinct instructions at which the instruction signal reaches 1.0.
INSTRUCTIONS_FOR_FULL = 4

#: Marks that end a question, including the ones other scripts use.
QUESTION_MARKS = ("?", "？", "؟")

#: Words that make a sentence a question. Matched as whole words, so these
#: only work for scripts that put spaces between words.
QUESTION_WORDS = (
    # English
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "whether",
    # Russian and other Cyrillic
    "почему",  # pochemu / why
    "зачем",  # zachem / what for
    "как",  # kak / how
    "какой",  # kakoy / which
    "что",  # chto / what
    "когда",  # kogda / when
    "где",  # gde / where
    "кто",  # kto / who
    # Arabic
    "لماذا",  # limadha / why
    "كيف",  # kayf / how
    "ماذا",  # madha / what
    "متى",  # mata / when
    "أين",  # ayna / where
    # Hindi and other Devanagari
    "क्यों",  # kyon / why
    "कैसे",  # kaise / how
    "क्या",  # kya / what
    "कब",  # kab / when
    "कहाँ",  # kahan / where
    # Greek
    "γιατί",  # giati / why
    "πώς",  # pos / how
    "τι",  # ti / what
    "πότε",  # pote / when
    # Hebrew
    "למה",  # lama / why
    "איך",  # eikh / how
    "מה",  # ma / what
    "מתי",  # matai / when
    "איפה",  # eifo / where
)

#: Phrases that make a sentence a question without a question word. Matched as
#: substrings, which is what a script written without spaces needs.
QUESTION_PHRASES = (
    # English
    "should i",
    "can i",
    "is it",
    "are there",
    "do i",
    # Japanese
    "なぜ",  # naze / why
    "どう",  # dou / how
    "どの",  # dono / which
    "いつ",  # itsu / when
    "どこ",  # doko / where
    "ですか",  # desu ka
    "ますか",  # masu ka
    # Chinese
    "为什么",  # weishenme / why (simplified)
    "為什麼",  # weishenme / why (traditional)
    "如何",  # ruhe / how
    "怎么",  # zenme / how (simplified)
    "怎麼",  # zenme / how (traditional)
    "什么",  # shenme / what (simplified)
    "什麼",  # shenme / what (traditional)
    "哪个",  # nage / which
    # Korean
    "왜",  # wae / why
    "어떻게",  # eotteoke / how
    "무엇",  # mueot / what
    "언제",  # eonje / when
    "어디",  # eodi / where
)

#: Substrings that say the answer needs thought rather than recall. English
#: first, then the same ideas in the other scripts this scorer reads.
REASONING_MARKERS = (
    # English
    "explain",
    "why",
    "analyz",
    "analys",
    "compare",
    "contrast",
    "trade-off",
    "tradeoff",
    "pros and cons",
    "prove",
    "derive",
    "justify",
    "evaluate",
    "critique",
    "implication",
    "step by step",
    "step-by-step",
    "reason",
    "root cause",
    "diagnose",
    "optimi",
    "design",
    "architect",
    "strategy",
    "recommend",
    "because",
    # Japanese
    "説明",  # setsumei / explain
    "なぜ",  # naze / why
    "比較",  # hikaku / compare
    "分析",  # bunseki / analyse
    "理由",  # riyuu / reason
    "証明",  # shoumei / prove
    "評価",  # hyouka / evaluate
    "推奨",  # suishou / recommend
    "段階的",  # dankaiteki / step by step
    "最適化",  # saitekika / optimise
    "設計",  # sekkei / design
    "トレードオフ",  # toreedoofu / trade-off
    # Chinese
    "解释",  # jieshi / explain (simplified)
    "解釋",  # jieshi / explain (traditional)
    "为什么",  # weishenme / why (simplified)
    "為什麼",  # weishenme / why (traditional)
    "比较",  # bijiao / compare (simplified)
    "证明",  # zhengming / prove (simplified)
    "證明",  # zhengming / prove (traditional)
    "评估",  # pinggu / evaluate (simplified)
    "評估",  # pinggu / evaluate (traditional)
    "推荐",  # tuijian / recommend (simplified)
    "推薦",  # tuijian / recommend (traditional)
    "优化",  # youhua / optimise (simplified)
    "優化",  # youhua / optimise (traditional)
    "设计",  # sheji / design (simplified)
    "原因",  # yuanyin / cause
    "逐步",  # zhubu / step by step
    "权衡",  # quanheng / trade-off (simplified)
    "權衡",  # quanheng / trade-off (traditional)
    # Korean
    "설명",  # seolmyeong / explain
    "비교",  # bigyo / compare
    "분석",  # bunseok / analyse
    "증명",  # jeungmyeong / prove
    "평가",  # pyeongga / evaluate
    "추천",  # chucheon / recommend
    "최적화",  # choejeokhwa / optimise
    "설계",  # seolgye / design
    "이유",  # iyu / reason
    "단계별",  # dangyebyeol / step by step
    # Russian and other Cyrillic
    "объясн",  # obyasn- / explain
    "почему",  # pochemu / why
    "сравн",  # sravn- / compare
    "анализ",  # analiz / analysis
    "доказ",  # dokaz- / prove
    "оцен",  # otsen- / evaluate
    "рекоменд",  # rekomend- / recommend
    "причин",  # prichin- / cause
    "оптимиз",  # optimiz- / optimise
    "пошагов",  # poshagov- / step by step
    "обоснов",  # obosnov- / justify
    "проектир",  # proektir- / design
    # Arabic
    "شرح",  # sharh / explain
    "لماذا",  # limadha / why
    "قارن",  # qaarin / compare
    "مقارن",  # muqaaran- / comparison
    "تحليل",  # tahlil / analysis
    "إثبات",  # ithbat / proof
    "تقييم",  # taqyim / evaluation
    "توصي",  # tawsiy- / recommend
    "سبب",  # sabab / reason
    "تصميم",  # tasmim / design
    # Hindi and other Devanagari
    "समझा",  # samjha- / explain
    "तुलना",  # tulna / compare
    "विश्लेषण",  # vishleshan / analysis
    "क्यों",  # kyon / why
    "सिद्ध",  # siddh / prove
    "मूल्यांकन",  # mulyankan / evaluate
    "सुझाव",  # sujhav / suggestion
    "कारण",  # karan / reason
    # Greek. Greek writes the stress accent on the stem vowel, so a stem is
    # listed both ways rather than being stripped of its accents.
    "εξηγ",  # exig- / explain
    "εξήγ",  # exig- / explain, stressed
    "γιατί",  # giati / why
    "συγκρ",  # sygkr- / compare
    "σύγκρ",  # sygkr- / compare, stressed
    "αναλυ",  # analy- / analyse
    "ανάλυ",  # analy- / analyse, stressed
    "αποδειξ",  # apodeix- / prove
    "απόδειξ",  # apodeix- / prove, stressed
    "αξιολογ",  # axiolog- / evaluate
    "αξιολόγ",  # axiolog- / evaluate, stressed
    "σχεδιασ",  # schedias- / design
    "σχεδίασ",  # schedias- / design, stressed
    "προτειν",  # protein- / recommend
    "πρότειν",  # protein- / recommend, stressed
    "βήμα βήμα",  # vima vima / step by step
    # Hebrew
    "הסבר",  # hesber / explain
    "למה",  # lama / why
    "השוו",  # hashvaa / compare
    "נתח",  # nateach / analyse
    "הוכח",  # hokhach / prove
    "המלץ",  # hamlets / recommend
    "סיבה",  # siba / reason
)

#: Substrings that say there is code in the prompt. Code is written in ASCII
#: whatever language surrounds it, so this list needs no translation.
CODE_MARKERS = (
    "```",
    "def ",
    "class ",
    "import ",
    "#include",
    "function",
    "=>",
    "();",
    "()",
    "{}",
    "};",
    "</",
    "/>",
    "#!/",
    "print(",
    "console.log",
    "traceback",
    "stack trace",
    "select ",
    "pip install",
    "npm ",
    "regex",
)

#: Verbs that open an instruction.
IMPERATIVE_VERBS = frozenset(
    (
        "add analyse analyze build calculate check compare compute convert "
        "create debug define describe design draft evaluate explain extract "
        "find fix generate give implement improve list make optimise optimize "
        "outline propose refactor remove rename replace review rewrite show "
        "sort suggest summarise summarize tell test translate update validate "
        "write"
    ).split()
)

#: Phrases that chain one more instruction onto a prompt. The spaced entries
#: are English; a script written without spaces needs them unspaced.
CHAIN_PHRASES = (
    " and then ",
    " after that ",
    " finally, ",
    " then also ",
    "そして",  # soshite / and then (ja)
    "その後",  # sono ato / after that (ja)
    "最後に",  # saigo ni / finally (ja)
    "然后",  # ranhou / and then (zh)
    "最后",  # zuihou / finally (zh)
    "затем",  # zatem / then (ru)
    "наконец",  # nakonets / finally (ru)
    "ثم",  # thumma / then (ar)
    "και μετά",  # kai meta / and then (el)
    "ואז",  # ve'az / and then (he)
    "ולבסוף",  # ulevasof / and finally (he)
    "उसके बाद",  # uske baad / after that (hi)
)

#: A run of characters from a script written without spaces: Chinese, the two
#: Japanese kana, and Hangul.
_CJK_RUN_RE = re.compile(
    "["
    "々〆぀-ヿ㐀-䶿一-鿿"
    "豈-﫿가-힯"
    "]+"
)
#: A word in any script that puts spaces between words.
_WORD_RE = re.compile("[^\\W_]+(?:['’][^\\W_]+)*", re.UNICODE)
#: Code points that belong to the Latin script.
_LATIN_LETTER_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),
    (0x1E00, 0x1EFF),
    (0x2C60, 0x2C7F),
    (0xA720, 0xA7FF),
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
#: Sentence ends: the ASCII ones need whitespace after them, the ones from
#: scripts written without spaces do not.
_SENTENCE_SPLIT_RE = re.compile(
    "(?<=[.!?])\\s+|(?<=[。！？۔।])\\s*|\\n+"
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Keep ``value`` inside ``[low, high]``."""
    return max(low, min(high, value))


def estimate_tokens(text: str) -> int:
    """Tokens a piece of text is worth, at ``CHARS_PER_TOKEN`` characters each."""
    return int(math.ceil(len(text) / float(CHARS_PER_TOKEN)))


def _spaced_words(text: str) -> List[str]:
    """Whole words from the part of ``text`` that is written with spaces."""
    return _WORD_RE.findall(_CJK_RUN_RE.sub(" ", text))


def count_words(text: str) -> int:
    """Words in ``text``, in any script.

    A script that separates words with spaces is counted word by word. Chinese,
    Japanese and Korean are written without spaces, so a run of them counts one
    word per ``CJK_CHARS_PER_WORD`` characters instead of collapsing to one.
    """
    words = len(_spaced_words(text))
    dense = sum(len(run) for run in _CJK_RUN_RE.findall(text))
    if dense:
        words += int(math.ceil(dense / float(CJK_CHARS_PER_WORD)))
    return words


def _is_latin_letter(char: str) -> bool:
    code = ord(char)
    for low, high in _LATIN_LETTER_RANGES:
        if low <= code <= high:
            return True
    return False


def non_latin_share(text: str) -> float:
    """Share of the letters in ``text`` that are not Latin, between 0 and 1."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for char in letters if not _is_latin_letter(char))
    return foreign / float(len(letters))


def _length_signal(words: int) -> float:
    return _clamp(words / float(WORDS_FOR_FULL_LENGTH))


def _question_hits(lowered: str, tokens: List[str]) -> int:
    hits = sum(lowered.count(mark) for mark in QUESTION_MARKS)
    hits += sum(1 for token in tokens if token in QUESTION_WORDS)
    hits += sum(1 for phrase in QUESTION_PHRASES if phrase in lowered)
    return hits


def _found_markers(lowered: str, markers: Tuple[str, ...]) -> List[str]:
    return [marker for marker in markers if marker in lowered]


#: An imperative verb opening a clause: at the start, after sentence punctuation, or
#: after a comma or semicolon, optionally behind "and", "then" or "also".
_CLAUSE_VERB_RE = re.compile(
    r"(?:^|[.;:!?]|,)\s*(?:and\s+|then\s+|also\s+|next\s+|finally,?\s+)*([a-z]+)",
    re.IGNORECASE,
)


def _count_instructions(prompt: str, lowered: str) -> int:
    """Distinct instructions: list items, imperative clauses and chains.

    Instructions are counted per CLAUSE, not per sentence. People pack several into
    one sentence - "Refactor the service, then write tests, explain the trade-offs,
    and outline the migration" is four requests with one full stop - and counting
    sentences alone scored that as a single instruction, leaving the signal at zero
    for exactly the multi-part prompts it exists to detect. The chain phrases only
    matched exact spacings such as " and then ", so ", then write" slipped past them
    too.
    """
    if not prompt.strip():
        return 0
    lines = prompt.splitlines()
    list_items = sum(1 for line in lines if _LIST_ITEM_RE.match(line))
    imperatives = sum(
        1 for match in _CLAUSE_VERB_RE.finditer(prompt)
        if match.group(1).lower() in IMPERATIVE_VERBS
    )
    chained = sum(lowered.count(phrase) for phrase in CHAIN_PHRASES)
    return max(1, list_items, imperatives) + min(chained, 3)


@dataclass(frozen=True)
class Complexity:
    """What ``complexity(prompt)`` found. Frozen, JSON-safe, self-explaining."""

    score: float
    band: str
    words: int
    characters: int
    estimated_tokens: int
    signals: Dict[str, float]
    reasons: Tuple[str, ...]
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """One human line, plain ASCII punctuation."""
        head = "{0} prompt (complexity {1:.2f})".format(self.band, self.score)
        if not self.characters:
            return head + ": nothing to weigh, the prompt is empty"
        parts = list(self.reasons) + list(self.warnings)
        if not parts:
            return head + ": {0} characters, nothing any signal recognises".format(
                self.characters
            )
        return head + ": " + "; ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict of everything above."""
        return {
            "score": self.score,
            "band": self.band,
            "words": self.words,
            "characters": self.characters,
            "estimated_tokens": self.estimated_tokens,
            "signals": dict(self.signals),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }

    def __str__(self) -> str:
        return self.summary()


def band_for(score: float) -> str:
    """The band a score falls in: simple, moderate or hard."""
    if score < SIMPLE_MAX:
        return "simple"
    if score < MODERATE_MAX:
        return "moderate"
    return "hard"


def _warnings_for(
    prompt: str, score: float, words: int, signals: Dict[str, float]
) -> Tuple[str, ...]:
    """Say so when the scorer could not read a prompt, instead of scoring 0.0.

    An empty prompt earns no warning: 0.0 is the right answer for it. So does a
    handful of words in any script - a greeting is simple everywhere. A prompt
    with real substance in a script whose markers are not in the lists does
    warn, and so does text that matched nothing at all.
    """
    if not prompt.strip():
        return ()
    unread = all(
        signals[name] == 0.0 for name in ("questions", "reasoning", "instructions")
    )
    if (
        unread
        and words >= WORDS_FOR_SCRIPT_WARNING
        and non_latin_share(prompt) >= NON_LATIN_SHARE
    ):
        return (
            "scored on length and structure only: this prompt is mostly in a "
            "script whose question and reasoning words are not in the marker "
            "lists, so the score is lower than the prompt may deserve",
        )
    if score == 0.0:
        return (
            "no signal recognised anything in this prompt, so it scored 0.00 "
            "and bands as simple",
        )
    return ()


def complexity(prompt: str) -> Complexity:
    """Score how hard ``prompt`` is to answer, between 0 and 1.

    An empty prompt is simple, not an error. A prompt that is not a string
    raises ``TypeError`` naming the type it got. A prompt in a script whose
    language markers this scorer does not carry is still scored on length and
    structure, and the result says so in ``warnings`` rather than quietly
    coming back as 0.0.
    """
    if not isinstance(prompt, str):
        raise TypeError(
            "prompt must be a string, got {0}".format(type(prompt).__name__)
        )

    lowered = prompt.lower()
    tokens = _spaced_words(lowered)
    words = count_words(prompt)

    question_hits = _question_hits(lowered, tokens)
    reasoning_found = _found_markers(lowered, REASONING_MARKERS)
    code_found = _found_markers(lowered, CODE_MARKERS)
    instructions = _count_instructions(prompt, lowered)

    code_signal = _clamp(len(code_found) / float(CODE_MARKERS_FOR_FULL))
    if "```" in prompt:
        code_signal = max(code_signal, 0.8)

    signals = {
        "length": round(_length_signal(words), 4),
        "questions": round(_clamp(question_hits / float(QUESTIONS_FOR_FULL)), 4),
        "reasoning": round(_clamp(len(reasoning_found) / float(REASONING_FOR_FULL)), 4),
        "code": round(code_signal, 4),
        "instructions": round(
            _clamp(max(0, instructions - 1) / float(INSTRUCTIONS_FOR_FULL - 1)), 4
        ),
    }
    score = round(
        sum(signals[name] * weight for name, weight in SIGNAL_WEIGHTS.items()), 4
    )

    reasons: List[str] = []
    if words:
        reasons.append("{0} word{1}".format(words, "" if words == 1 else "s"))
    if question_hits:
        reasons.append(
            "{0} question signal{1}".format(
                question_hits, "" if question_hits == 1 else "s"
            )
        )
    if reasoning_found:
        reasons.append("reasoning words: " + ", ".join(reasoning_found[:4]))
    if code_found:
        reasons.append("code markers: " + ", ".join(repr(m) for m in code_found[:3]))
    if instructions > 1:
        reasons.append("{0} separate instructions".format(instructions))

    warnings = _warnings_for(prompt, score, words, signals)
    for note in warnings:
        logger.warning("%s", note)

    return Complexity(
        score=score,
        band=band_for(score),
        words=words,
        characters=len(prompt),
        estimated_tokens=estimate_tokens(prompt),
        signals=signals,
        reasons=tuple(reasons),
        warnings=warnings,
    )
