# document-memory

Give an application a persistent, searchable memory of documents and conversations: one SQLite
file, real BM25 ranking, and a `context()` call that hands you the best matches already packed
to a prompt budget.

## Install

```
pip install document-memory
```

## Quickstart

```python
from document_memory import Memory

with Memory("notes.db") as memory:                       # the file is created on demand
    memory.add("Solar panels turn sunlight into electricity.", source="guide.md")
    memory.add("Wind turbines turn moving air into electricity.", source="guide.md")
    memory.remember("user", "which one works on a cloudy roof?")
    print(memory.search("solar")[0].summary())
    print(memory.context("electricity", budget=40))
```

Reopen `Memory("notes.db")` tomorrow, or from another process, and everything is still there.

## What it does

- **Persistent by default.** One SQLite file holds the documents, the conversation turns, the
  metadata and the BM25 index. Nothing is kept only in RAM, so a restart loses nothing.
  `Memory()` with no path gives a private in-memory store instead, which is what tests want.
- **Safe for several processes at once.** The file is opened in WAL mode with a
  `BUSY_TIMEOUT`-second busy timeout and every write runs inside `BEGIN IMMEDIATE`. Two
  workers writing to the same store queue behind each other instead of raising
  `sqlite3.OperationalError: database is locked`.
- **Ranking is real Okapi BM25**, `k1 = 1.5` and `b = 0.75`, with the non-negative Lucene idf.
  A rare query term outranks a common one; a long document is not rewarded for merely being
  long. The constants are importable as `BM25_K1` and `BM25_B`, and the test suite asserts the
  rare-term property rather than trusting it.
- **Vectors are optional and blended, not bolted on.** Pass `embed=` and every memory is stored
  with its vector; search then scores
  `(1 - VECTOR_WEIGHT) * bm25_normalised + VECTOR_WEIGHT * cosine` with `VECTOR_WEIGHT = 0.5`, a
  documented constant you can change per store with `memory.vector_weight = 0.3`. Without
  `embed` no vectors are stored and nothing slows down. numpy is used here and nowhere else.
- **Every result explains itself.** A `Hit` carries `score` (normalised, best match 1.0), `bm25`
  (the raw, comparable value), `similarity`, and `terms` - the query terms that memory actually
  contained. `hit.why()` spells it out in one line, `hit.summary()` adds a text preview, and
  `hit.to_dict()` is JSON-safe.
- **`context(query, budget=...)` never exceeds the budget.** Memories are packed in relevance
  order and the header and blank lines are counted too, so `counter(context) <= budget` holds
  exactly. The default unit is whitespace words; pass `counter=your_tokeniser` to budget in real
  tokens. A budget too small for a whole memory truncates one rather than returning nothing.
- **Conversations are first-class.** `remember(role, content)` stores a turn with the role in its
  metadata, `history(n)` returns the last `n` turns *oldest first* so they replay in order, and
  turns rank alongside documents in ordinary search.
- **Metadata round-trips exactly**, nested dicts, lists and unicode included - it is stored as
  JSON with `ensure_ascii=False`. `where=` filters on it: `{"project": "roof"}`, a list for "any
  of", `{"user.name": "ada"}` for a nested key, or a callable for anything else.
- **Namespaces** are independent compartments inside one file - one per user, per project or per
  tenant - and they never see each other's memories.
- **The dull cases are handled.** Search on an empty store returns `[]`. `delete()` of an unknown
  id returns `False`. Re-adding the same id replaces that memory instead of duplicating it. All
  of these are covered by tests.
- **Fast enough to ignore.** 10,000 documents search in roughly 2 ms, because the index is a
  postings table and only the rows containing a query term are ever touched.

## API

### `Memory(path=None, *, namespace="default", embed=None)`

`path` is the SQLite file, created on demand along with its parent folders; `None` means a
private in-memory store. `embed(list_of_texts) -> list_of_vectors` switches on the hybrid
ranking. Usable as a context manager.

| Method | Returns | What it does |
| --- | --- | --- |
| `add(text, *, id=None, metadata=None, source=None, timestamp=None)` | `str` | Stores one memory, returns its id. A repeated `id` replaces in place. |
| `add_many(items)` | `list[str]` | Same, one transaction, one `embed` call. Items are strings, dicts with a `text` key, or `Record`s. |
| `search(query, *, k=5, where=None, since=None)` | `Hits` (a `list` of `Hit`) | BM25, blended with cosine when `embed` is set. |
| `get(id)` | `Record` or `None` | One memory by id. |
| `delete(id)` | `bool` | `False` if the id was not there. |
| `update(id, text=None, metadata=None)` | `Record` or `None` | Metadata is merged key by key; `None` drops a key. Re-embeds when the text changes. |
| `recent(n=10, *, source=None)` | `Records` | Newest first. |
| `count()` | `int` | Memories in this namespace. |
| `clear(namespace=None)` | `int` | How many went. `"*"` empties the whole file. |
| `remember(role, content, **meta)` | `str` | A conversation turn; `role` is kept in the metadata. |
| `history(n=20)` | `Turns` | The last `n` turns, oldest first. |
| `context(query, *, budget=2000, counter=None)` | `str` | Best matches packed to the budget, ready to paste. |
| `summary()` | `str` | One line describing the store. |
| `close()` | `None` | Closes the handle; idempotent. |

### Result objects

`Hit(id, text, score, metadata, source, ...)` also carries `bm25`, `similarity`, `terms`,
`when`, `why()`, `summary()`, `to_dict()` and `to_record()`. `Record` and `Turn` are the same
shape without the ranking. `Hits`, `Records` and `Turns` are real `list` subclasses - `result ==
[]` and `result[0]` behave exactly as you expect - with a `summary()` and a `to_dict()` that
describe the whole set.

### Constants

`VECTOR_WEIGHT` (0.5), `VECTOR_FLOOR` (0.15, the cosine below which a memory sharing no query
term is not a match), `BUSY_TIMEOUT` (30.0 seconds), `BM25_K1` (1.5), `BM25_B` (0.75),
`CONTEXT_HEADER`. `tokenize(text)` is exported so you can see exactly what gets indexed.

## CLI

```
document-memory notes.db add "Solar panels turn light into power." --source guide.md
document-memory notes.db search "solar" -k 3
document-memory notes.db search "solar" --where '{"project": "roof"}' --json
document-memory notes.db context "solar" --budget 120
document-memory notes.db remember user "how do panels do in winter?"
document-memory notes.db history -n 10
echo "panels on the roof" | document-memory notes.db add -
```

`STORE` is the file, created on demand. With no command the store describes itself. `--json`
prints `to_dict()`, `--output PATH` also writes the result to a file, `--namespace` picks the
compartment. Output is UTF-8 whatever it is piped into. `document-memory --help` lists
everything.

## License

MIT
