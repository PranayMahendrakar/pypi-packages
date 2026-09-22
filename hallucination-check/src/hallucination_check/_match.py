"""Finding the best-matching span of the sources for one claim.

The sources are split into sentences once; an inverted index shortlists the few
sentences worth scoring, and only those get the full lexical comparison. That is what
keeps a few thousand retrieved passages under a second.
"""
from __future__ import annotations

from math import log
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._text import TextVec, cosine, coverage, stem, token_spans, tokenize, vectorize

__all__ = ["Span", "SourceIndex", "lexical_score"]

# Weights of the three lexical signals. Coverage carries the most weight because a
# claim is short and the span it is checked against is not.
W_CHAR = 0.30
W_WORD = 0.25
W_COVER = 0.45

# A span is often far longer than the claim being checked against it: one long source
# sentence, or three of them joined. The character and word cosines are symmetric, so
# comparing a short claim with all of that penalizes the claim for text it never set out
# to cover -- a faithful paraphrase of one clause of a 200-word sentence scores lower and
# lower as the rest of the sentence grows, until a correct answer is called a fabrication.
# So a claim is also compared with the stretch of the span, its own size, that holds the
# most of it, and keeps whichever score is better. A window only ever removes surrounding
# text: it cannot invent overlap that is not in the span, so nothing becomes supported
# that the span did not already say.
MIN_WINDOW_TOKENS = 4
WINDOW_RATIO = 2


class Span:
    """A run of consecutive sentences from one source."""

    __slots__ = ("source", "start", "end")

    def __init__(self, source: int, start: int, end: int) -> None:
        self.source = source
        self.start = start
        self.end = end

    @property
    def key(self) -> Tuple[int, int]:
        return (self.start, self.end)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Span(source=%d, %d..%d)" % (self.source, self.start, self.end)


def lexical_score(claim: TextVec, span: TextVec) -> float:
    """Blend character n-gram cosine, word n-gram cosine and content coverage."""
    char = cosine(claim.char, claim.char_norm, span.char, span.char_norm)
    word = cosine(claim.word, claim.word_norm, span.word, span.word_norm)
    cover = coverage(claim, span)
    return max(0.0, min(1.0, W_CHAR * char + W_WORD * word + W_COVER * cover))


class SourceIndex:
    """Sentence-level index over the sources, with cached span vectors."""

    def __init__(
        self,
        sources: Sequence[Tuple[str, str]],
        *,
        char_n: int = 4,
        top_k: int = 25,
        window: int = 3,
    ) -> None:
        self.char_n = char_n
        self.top_k = max(1, int(top_k))
        self.window = max(1, int(window))
        self.source_ids: List[str] = [sid for sid, _text in sources]
        self.sentences: List[str] = []
        self.sent_source: List[int] = []
        postings: Dict[str, List[int]] = {}
        from ._text import split_sentences  # local import keeps the module graph flat

        for position, (_sid, text) in enumerate(sources):
            pieces = split_sentences(text) or ([text.strip()] if text.strip() else [])
            for piece in pieces:
                index = len(self.sentences)
                self.sentences.append(piece)
                self.sent_source.append(position)
                keys = set()
                for token in tokenize(piece):
                    keys.add(token)
                    keys.add(stem(token))
                for key in keys:
                    postings.setdefault(key, []).append(index)

        self.n_sentences = len(self.sentences)
        total = float(self.n_sentences)
        self._postings: Dict[str, np.ndarray] = {}
        self._idf: Dict[str, float] = {}
        for key, ids in postings.items():
            self._postings[key] = np.asarray(ids, dtype=np.int64)
            self._idf[key] = log(1.0 + total / (1.0 + len(ids)))
        self._vec_cache: Dict[Tuple[int, int], TextVec] = {}
        self._text_cache: Dict[Tuple[int, int], str] = {}
        self._token_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]] = {}
        self._window_cache: Dict[str, TextVec] = {}

    def __len__(self) -> int:
        return self.n_sentences

    def span_text(self, span: Span) -> str:
        """The source text covered by ``span``."""
        cached = self._text_cache.get(span.key)
        if cached is None:
            cached = " ".join(self.sentences[span.start : span.end + 1])
            self._text_cache[span.key] = cached
        return cached

    def span_vec(self, span: Span) -> TextVec:
        """A cached :class:`TextVec` for ``span``."""
        cached = self._vec_cache.get(span.key)
        if cached is None:
            from ._facts import number_tokens

            text = self.span_text(span)
            cached = vectorize(text, char_n=self.char_n, extra_tokens=number_tokens(text))
            self._vec_cache[span.key] = cached
        return cached

    def span_tokens(self, span: Span) -> List[Tuple[str, int, int]]:
        """Tokens of ``span`` with their offsets into :meth:`span_text`."""
        cached = self._token_cache.get(span.key)
        if cached is None:
            cached = token_spans(self.span_text(span))
            self._token_cache[span.key] = cached
        return cached

    def window_text(self, span: Span, claim: TextVec) -> Optional[str]:
        """The stretch of ``span``, the size of ``claim``, that holds most of it.

        ``None`` when the span is not long enough for the size difference to matter, or
        when no word of the claim occurs in it at all.
        """
        size = len(claim.tokens)
        if size < MIN_WINDOW_TOKENS:
            return None
        tokens = self.span_tokens(span)
        if len(tokens) < WINDOW_RATIO * size:
            return None
        wanted = claim.match_set
        hits = [
            1 if (token.casefold() in wanted or stem(token.casefold()) in wanted) else 0
            for token, _start, _end in tokens
        ]
        running = sum(hits[:size])
        best, best_at = running, 0
        for position in range(size, len(hits)):
            running += hits[position] - hits[position - size]
            if running > best:
                best, best_at = running, position - size + 1
        if best <= 0:
            return None
        text = self.span_text(span)
        return text[tokens[best_at][1] : tokens[best_at + size - 1][2]]

    def window_vec(self, span: Span, claim: TextVec) -> Optional[TextVec]:
        """A cached :class:`TextVec` for :meth:`window_text`, or ``None``."""
        text = self.window_text(span, claim)
        if text is None:
            return None
        cached = self._window_cache.get(text)
        if cached is None:
            from ._facts import number_tokens

            cached = vectorize(text, char_n=self.char_n, extra_tokens=number_tokens(text))
            self._window_cache[text] = cached
        return cached

    def source_id(self, span: Span) -> str:
        """The id of the source ``span`` came from."""
        return self.source_ids[span.source]

    def candidates(self, claim: TextVec) -> List[Span]:
        """Spans worth scoring for ``claim``, shortlisted by weighted token overlap."""
        if self.n_sentences == 0:
            return []
        scores = np.zeros(self.n_sentences, dtype=np.float64)
        hit = False
        for token in set(claim.keys):
            for key in (token, stem(token)):
                ids = self._postings.get(key)
                if ids is not None:
                    scores[ids] += self._idf[key]
                    hit = True
        if not hit:
            return []
        k = min(self.top_k, self.n_sentences)
        if k < self.n_sentences:
            picked = np.argpartition(-scores, k - 1)[:k]
        else:
            picked = np.arange(self.n_sentences)
        seeds = [int(i) for i in picked if scores[i] > 0.0]
        seeds.sort(key=lambda i: (-scores[i], i))

        return self.windows(seeds)

    def windows(self, seeds: Sequence[int]) -> List[Span]:
        """Spans of up to ``window`` sentences built around each seed sentence."""
        shapes = [(0, 0)]
        if self.window >= 2:
            shapes += [(0, 1), (-1, 0)]
        if self.window >= 3:
            shapes.append((-1, 1))

        spans: List[Span] = []
        seen = set()
        for seed in seeds:
            source = self.sent_source[seed]
            for back, ahead in shapes:
                start, end = seed + back, seed + ahead
                if start < 0 or end >= self.n_sentences:
                    continue
                if self.sent_source[start] != source or self.sent_source[end] != source:
                    continue
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(Span(source, start, end))
        return spans
