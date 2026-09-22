"""Turn a string or a ``.txt`` / ``.md`` / ``.html`` path into a :class:`Source`.

A ``Source`` is the exact text that will be chunked plus the structure found in
it: heading offsets (so a chunk can carry its heading trail) and *atomic* spans
-- fenced code blocks and tables -- that must never be cut in half.
"""
from __future__ import annotations

import bisect
import os
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

TEXT_SUFFIXES = frozenset({".txt", ".text"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdown"})
HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
READABLE_SUFFIXES = TEXT_SUFFIXES | MARKDOWN_SUFFIXES | HTML_SUFFIXES

SOURCE_KINDS = ("auto", "text", "markdown", "html")

_MD_HEADING = re.compile(r"^([ \t]{0,3})(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", re.M)
_LOOKS_HTML_OPEN = re.compile(
    r"<\s*(?:html|body|div|p|h[1-6]|table|pre|ul|ol|li|section|article|span)\b", re.I
)
_LOOKS_HTML_CLOSE = re.compile(
    r"</\s*(?:html|body|div|p|h[1-6]|table|pre|ul|ol|li|section|article|span)\s*>", re.I
)


def _merge_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


@dataclass
class Source:
    """The text to chunk, plus the structure discovered in it."""

    text: str
    kind: str = "text"
    headings: List[Tuple[int, int, str]] = field(default_factory=list)
    atomic: List[Tuple[int, int]] = field(default_factory=list)
    origin: Optional[str] = None

    def __post_init__(self) -> None:
        self.headings = sorted(self.headings)
        self.atomic = _merge_spans(self.atomic)
        self._path_offsets: List[int] = []
        self._paths: List[Tuple[str, ...]] = []
        stack: List[Tuple[int, str]] = []
        for offset, level, title in self.headings:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            self._path_offsets.append(offset)
            self._paths.append(tuple(t for _, t in stack))

    def heading_path_at(self, offset: int) -> Tuple[str, ...]:
        """The heading trail in effect at ``offset`` (empty before the first heading)."""
        if not self._path_offsets:
            return ()
        i = bisect.bisect_right(self._path_offsets, offset) - 1
        return self._paths[i] if i >= 0 else ()

    @property
    def heading_offsets(self) -> List[int]:
        """Character offsets where a heading starts."""
        return [offset for offset, _, _ in self.headings]


# --------------------------------------------------------------------------- #
# markdown structure
# --------------------------------------------------------------------------- #
def _line_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        spans.append((pos, pos + len(line), line))
        pos += len(line)
    return spans


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _is_table_rule(line: str) -> bool:
    stripped = line.strip()
    if not stripped or "-" not in stripped or "|" not in stripped:
        return False
    return set(stripped) <= set("|:- \t")


def markdown_atomic_spans(text: str) -> List[Tuple[int, int]]:
    """Spans of fenced code blocks and pipe tables, which are never split."""
    lines = _line_spans(text)
    spans: List[Tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        start, _end, raw = lines[i]
        fence = raw.strip()[:3]
        if fence in ("```", "~~~"):
            j = i + 1
            while j < n and not lines[j][2].strip().startswith(fence):
                j += 1
            block_end = lines[j][1] if j < n else lines[n - 1][1]
            spans.append((start, block_end))
            i = j + 1 if j < n else n
            continue
        if _is_table_line(raw):
            j = i
            while j < n and _is_table_line(lines[j][2]):
                j += 1
            if j - i >= 2 and any(_is_table_rule(lines[k][2]) for k in range(i, j)):
                spans.append((start, lines[j - 1][1]))
                i = j
                continue
        i += 1
    return _merge_spans(spans)


def markdown_headings(
    text: str, atomic: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int, str]]:
    """ATX headings (``# Title``) outside code fences, as ``(offset, level, title)``."""
    out: List[Tuple[int, int, str]] = []
    for match in _MD_HEADING.finditer(text):
        start = match.start()
        if any(s <= start < e for s, e in atomic):
            continue
        title = match.group(3).strip()
        out.append((start, len(match.group(2)), title or "(untitled)"))
    return out


# --------------------------------------------------------------------------- #
# html structure
# --------------------------------------------------------------------------- #
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_ATOMIC_TAGS = frozenset({"pre", "table"})
_PARA_TAGS = frozenset(
    {
        "p", "div", "section", "article", "blockquote", "pre", "table", "ul", "ol",
        "dl", "figure", "form", "main", "header", "footer", "aside", "nav", "hr",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)
_LINE_TAGS = frozenset(
    {"li", "tr", "dt", "dd", "td", "th", "figcaption", "caption", "option"}
)
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_SPACE_RUN = re.compile(r"\s+")


class _HTMLToText(HTMLParser):
    """Extract readable text from HTML while remembering headings and code/tables."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._len = 0
        self._tail = ""
        self._skip = 0
        self._atomic_depth = 0
        self._atomic_start = 0
        self.atomic: List[Tuple[int, int]] = []
        self.headings: List[Tuple[int, int, str]] = []
        self._heading: Optional[Tuple[int, int]] = None
        self._title: List[str] = []

    # -- output helpers -------------------------------------------------- #
    def _write(self, piece: str) -> None:
        if not piece:
            return
        self._parts.append(piece)
        self._len += len(piece)
        self._tail = (self._tail + piece)[-8:]

    def _break(self, count: int) -> None:
        if self._len == 0:
            return
        have = len(self._tail) - len(self._tail.rstrip("\n"))
        if have < count:
            self._write("\n" * (count - have))

    # -- parser hooks ---------------------------------------------------- #
    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _ATOMIC_TAGS:
            if self._atomic_depth == 0:
                self._break(2)
                self._atomic_start = self._len
            self._atomic_depth += 1
            return
        if tag == "br":
            self._write("\n")
            return
        if tag in _PARA_TAGS:
            self._break(2)
        elif tag in _LINE_TAGS:
            self._break(1)
        if tag in _HEADING_TAGS:
            self._heading = (int(tag[1]), self._len)
            self._title = []

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        if tag == "br" and not self._skip:
            self._write("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in _ATOMIC_TAGS:
            if self._atomic_depth:
                self._atomic_depth -= 1
                if self._atomic_depth == 0 and self._len > self._atomic_start:
                    self.atomic.append((self._atomic_start, self._len))
            return
        if tag in _HEADING_TAGS and self._heading is not None:
            level, start = self._heading
            title = _SPACE_RUN.sub(" ", "".join(self._title)).strip()
            self.headings.append((start, level, title or "(untitled)"))
            self._heading = None
            self._title = []
        if tag in _PARA_TAGS or tag in _LINE_TAGS:
            self._break(1)

    def handle_data(self, data: str) -> None:
        if self._skip or not data:
            return
        if self._atomic_depth:
            self._write(data)
            return
        collapsed = _SPACE_RUN.sub(" ", data)
        if collapsed == " " and (self._len == 0 or self._tail.endswith(("\n", " "))):
            return
        if self._heading is not None:
            self._title.append(collapsed)
        self._write(collapsed)

    def result(self) -> Tuple[str, List[Tuple[int, int, str]], List[Tuple[int, int]]]:
        return "".join(self._parts), self.headings, self.atomic


def html_to_source(markup: str, origin: Optional[str] = None) -> Source:
    """Extract text from HTML; ``<pre>`` and ``<table>`` become atomic spans."""
    parser = _HTMLToText()
    parser.feed(markup)
    parser.close()
    text, headings, atomic = parser.result()
    return Source(text=text, kind="html", headings=headings, atomic=atomic, origin=origin)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def looks_like_html(text: str) -> bool:
    """True when a bare string is clearly HTML markup rather than prose."""
    return bool(_LOOKS_HTML_OPEN.search(text) and _LOOKS_HTML_CLOSE.search(text))


# Characters no file name may hold on Windows, so a string carrying one is prose.
_ILLEGAL_IN_NAMES = '<>"|?*'


def path_shaped(value: str) -> bool:
    """True when a string with a document suffix reads as a path, not as prose.

    Only whitespace-free, filename-legal strings qualify, so a sentence that
    happens to end in ``.md`` and a bare URL are both still chunked as text.
    """
    if not value or len(value) > 260:
        return False
    if any(ch.isspace() for ch in value):
        return False
    if "://" in value:
        return False
    return not any(ch in value for ch in _ILLEGAL_IN_NAMES)


def looks_like_path(value: object) -> bool:
    """True when ``value`` should be read from disk instead of chunked as text.

    A string shaped like a document path counts even when the file is missing,
    so a typo or a wrong working directory raises ``FileNotFoundError`` instead
    of silently indexing the file name as though it were the document. This is
    what passing the same path as a ``Path`` has always done.
    """
    if isinstance(value, Path) or (
        isinstance(value, os.PathLike) and not isinstance(value, str)
    ):
        return True
    if not isinstance(value, str):
        return False
    if not value or len(value) > 4096 or any(ch in value for ch in "\n\r\x00"):
        return False
    if Path(value).suffix.lower() not in READABLE_SUFFIXES:
        return False
    try:
        if os.path.isfile(value):
            return True
    except (OSError, ValueError):  # pragma: no cover - exotic paths
        return False
    return path_shaped(value)


def read_document(path: object) -> Tuple[str, str, str]:
    """Read ``path`` and return ``(text, kind, origin)``. A UTF-8 BOM is stripped."""
    p = Path(os.fspath(path))
    suffix = p.suffix.lower()
    if suffix not in READABLE_SUFFIXES:
        raise ValueError(
            f"{p.name!r} is not a supported document: expected one of "
            + ", ".join(sorted(READABLE_SUFFIXES))
        )
    if not p.is_file():
        raise FileNotFoundError(f"{os.fspath(path)!r} does not exist")
    raw = p.read_text(encoding="utf-8-sig", errors="replace")
    if suffix in HTML_SUFFIXES:
        kind = "html"
    elif suffix in MARKDOWN_SUFFIXES:
        kind = "markdown"
    else:
        kind = "text"
    return raw, kind, str(p)


def load_source(text: object, kind: str = "auto") -> Source:
    """Build a :class:`Source` from a string or a ``.txt`` / ``.md`` / ``.html`` path.

    ``kind`` is one of ``"auto"``, ``"text"``, ``"markdown"``, ``"html"``.
    """
    if kind not in SOURCE_KINDS:
        raise ValueError(
            f"source must be one of {', '.join(SOURCE_KINDS)}, got {kind!r}"
        )
    origin: Optional[str] = None
    detected: Optional[str] = None
    if looks_like_path(text):
        raw, detected, origin = read_document(text)
    elif isinstance(text, str):
        raw = text
    else:
        raise TypeError(
            "text must be a string or a path to a .txt, .md or .html file, "
            f"got {type(text).__name__}"
        )
    if kind == "auto":
        kind = detected or ("html" if looks_like_html(raw) else "text")
    if kind == "html":
        return html_to_source(raw, origin=origin)
    atomic = markdown_atomic_spans(raw)
    headings = markdown_headings(raw, atomic)
    if kind == "text":
        kind = "markdown" if (headings or atomic) else "text"
    return Source(text=raw, kind=kind, headings=headings, atomic=atomic, origin=origin)
