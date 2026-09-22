# rag-chunker

Split documents for retrieval at meaning boundaries instead of fixed token counts, and get
back a result that tells you exactly what it did.

## Install

```
pip install rag-chunker
```

## Quickstart

```python
import rag_chunker

text = "Solar panels turn light into power. Rooftop arrays are the common case. " * 4 \
    + "Cats sleep about sixteen hours a day. They hunt at dawn and at dusk. " * 4
result = rag_chunker.chunk(text, size=40, overlap=8)
print(result.summary())
print(result.n_chunks, "chunks;", result.to_list()[0][:50], "...")
assert result.reassemble() == text   # chunks minus overlap rebuild the source exactly
```

`rag_chunker.chunk("guide.md")` works the same way: pass a string, or a path to a `.txt`,
`.md` or `.html` file. A value shaped like a document path is always read from disk, so a typo
raises `FileNotFoundError` naming it rather than quietly indexing the file name as the document -
and a `str` and a `pathlib.Path` behave identically. Prose that merely ends in `.md` (it has
spaces, or it is a URL) is still chunked as text.

## What it does

- **`semantic`** (default) turns every sentence into a hashed bag-of-words vector, measures the
  similarity across each sentence boundary using a two-sentence window on each side, and cuts at
  the *valleys* - the local minima below the 30th percentile of the document. Sentence bounds are
  always respected. No model, no download, no network: `numpy` is the only dependency.
- **`structural`** cuts on markdown (`# Title`) or HTML (`<h2>`) headings and hands every chunk its
  heading trail, so a chunk still knows which section it came from.
- **`sentence`** packs whole sentences up to `size`.
- **`fixed`** uses fixed windows, provided for comparison with the others.
- **`recursive`** takes paragraphs, falling back to sentences, then to words, whichever fits.
- **Nothing is ever dropped.** Chunk boundaries only land on unit boundaries that tile the source
  text end to end, so `result.reassemble() == result.text` for every method and every setting.
  `result.summary()` re-checks it each time it prints.
- **Fenced code blocks and markdown tables are never split down the middle**, and neither are
  `<pre>` or `<table>` in HTML: they are atomic, even when one is larger than `size`.
- **`size` and `overlap` are words by default and tokens when you pass `counter`.** Which one is in
  use is reported as `result.unit`, so the number is never ambiguous.
- Overlap is trimmed back to a whole sentence or word boundary, so a chunk never starts mid-word,
  and it only ever repeats the chunk directly before it - no chunk ever contains another, even when
  a page is a run of sections far shorter than `overlap` (an API reference, an FAQ, a changelog).
- **A heuristic that finds nothing says so.** When a document has no similarity valley to cut at
  (every boundary equally similar, or no two sentences sharing a word), `semantic` falls back to
  packing whole sentences and records why in `result.warnings`, `result.summary()` and `to_dict()`.
  `structural` does the same when a document has no headings.
- Deterministic: the word hash is `zlib.crc32`, so the same input gives the same chunks in every
  process and on every machine.
- Unicode is handled throughout, including CJK sentence enders (`。`, `！`, `？`).

## Which method to use

`recursive` is the default: paragraph, then sentence, then word. It is predictable,
never splits a word, and always reproduces the input exactly.

`semantic` compares word overlap between neighbouring sentences and cuts where it dips.
Be aware of what that can and cannot do. Two passages about genuinely different subjects
are only separated when they use different words, and ordinary prose often does not
repeat its own vocabulary enough for the dip to land in the right place. Measured on a
ten-sentence document with two clearly different topics, word overlap put the boundary
one sentence early. It is a useful heuristic on documents whose sections have distinct
vocabulary, not a meaning detector.

To get real semantics, pass an embedding model:

```python
chunks = rag_chunker.chunk(text, method="semantic", embed=my_model.encode)
```

`embed` takes the list of sentences and returns one vector per sentence. With a real
model the same ten-sentence document splits at exactly the right place. This package
does not depend on any embedding library, so you choose the model.

`structural` splits on markdown or HTML headings and keeps the heading trail on each
chunk. `sentence` packs whole sentences. `fixed` is there for comparison.

## API

```python
rag_chunker.chunk(text, *, size=512, overlap=64, method="recursive", counter=None, metadata=None) -> ChunkResult
rag_chunker.chunk_documents(docs, **kw) -> list[ChunkResult]
```

`text` is a string or a path to a `.txt` / `.md` / `.html` file; a path that does not exist raises
`FileNotFoundError`, whether it was given as a `str` or a `Path`. `method` is one of `semantic`,
`structural`, `sentence`, `fixed`, `recursive`. `counter` is any `callable(str) -> int` - pass your
tokenizer and both `size` and `overlap` are counted in its tokens, with no dependency on it.
`metadata` is a dict copied onto every chunk.

`docs` is a list of strings, paths, or `{"text": ..., "metadata": {...}}` dicts.

```python
import rag_chunker

rag_chunker.chunk(md, method="structural")                   # heading-aware
rag_chunker.chunk(text, size=256, counter=len)               # size counted by your callable
rag_chunker.chunk(text, metadata={"doc_id": 7})              # carried on every chunk
rag_chunker.chunk_documents(["a.md", "b.md"], method="recursive")
```

`Chunker(size=512, overlap=64, method="recursive", counter=None, metadata=None, *, sensitivity=30.0,
min_fill=0.5, source="auto")` is the class underneath, with `.chunk(text, metadata=None)` and
`.chunk_documents(docs)`. `sensitivity` is the percentile of similarity valleys treated as topic
shifts (higher splits more), `min_fill` is the fraction of `size` a chunk must reach before a topic
shift may end it, and `source` forces `text` / `markdown` / `html` parsing.

`ChunkResult`

- `.chunks` - `list[Chunk]`; the result is also iterable, indexable and `len()`-able
- `.n_chunks`, `.mean_size`, `.size_distribution` (count / min / p25 / median / p75 / max / mean / std)
- `.unit` - `"words"` or `"tokens"`, whichever `size` and `overlap` were counted in
- `.text` - the exact text that was chunked (for HTML, the text extracted from the markup)
- `.reassemble()` - the chunk bodies minus overlap, joined; equals `.text`
- `.reproduces_source` - the same check as a bool
- `.to_list()` - the chunk texts as `list[str]`; `.texts` is the same thing
- `.to_dict()` - JSON-safe dict of everything, chunks included
- `.summary()` - human-readable report
- `.method`, `.size`, `.overlap`, `.source_kind`, `.origin`, `.metadata`, `.n_words`

`Chunk`

- `.text`, `.index`, `.start`, `.end` - `text == source[start:end]`, always
- `.tokens` - the chunk's size in `result.unit`; `.n_chars` in characters
- `.heading_path` - tuple of headings above this chunk, e.g. `("Guide", "Setup")`
- `.overlap_with` - indices of earlier chunks whose text this chunk repeats
- `.metadata` - the dict you passed
- `.to_dict()`

Errors are plain `ValueError`s: `size=0` and negative sizes are rejected, and an `overlap` at least
as large as `size` is rejected with both numbers named.

## CLI

```
rag-chunker INPUT [--size 512] [--overlap 64] [--method semantic] [--source auto]
            [--sensitivity 30] [--metadata JSON] [--json] [--text] [--show N] [--output PATH]
```

- `rag-chunker guide.md` prints the summary and a preview of the first few chunks.
- `rag-chunker guide.md --method structural --json` prints `to_dict()` as JSON.
- `rag-chunker guide.md --text` prints the chunk texts with a marker line between them.
- `cat notes.txt | rag-chunker -` reads standard input.
- `--output PATH` writes JSON (`.json`), one chunk per line (`.jsonl`), or the chunk texts.

Output is written as UTF-8 even when piped or redirected, so non-ASCII documents never raise
`UnicodeEncodeError`.

## License

MIT
