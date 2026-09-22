"""The ``Memory`` class: a persistent, searchable store of documents and turns.

One object owns one SQLite file and one namespace inside it::

    with Memory("notes.db") as memory:
        memory.add("Solar panels turn light into power.", source="guide.md")
        print(memory.search("solar")[0].text)

Ranking is Okapi BM25 (see :mod:`document_memory._text` for the constants).
Pass ``embed=`` and the same search also compares vectors; the two halves are
averaged, and :data:`VECTOR_WEIGHT` says with what weight.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from . import _store, _text, _vectors
from ._results import Hit, Hits, Record, Records, Turn, Turns

logger = logging.getLogger(__name__)

BUSY_TIMEOUT = 30.0
"""Seconds a write waits for another process's lock before giving up.

SQLite's default is 0, which is what produces ``database is locked`` the
instant two processes collide.  Thirty seconds is long enough that ordinary
contention simply queues.
"""

VECTOR_WEIGHT = 0.5
"""How much of a hybrid score comes from the vectors, in ``[0, 1]``.

``score = (1 - VECTOR_WEIGHT) * bm25_normalised + VECTOR_WEIGHT * cosine``

The default 0.5 is an even split: neither half can carry a result on its own.
Change it per store with ``memory.vector_weight = 0.3`` (more lexical) or
``0.8`` (more semantic).  It has no effect without ``embed=``.
"""

VECTOR_FLOOR = 0.15
"""Cosine below which a memory containing no query term is not a match at all.

Without a floor, every memory in the store would be a hit in hybrid mode,
because every pair of vectors has some similarity.  A memory that does share a
query term is always considered, however weak its vector.
"""

CONTEXT_HEADER = "Relevant memories for {query!r}:"
"""First line of :meth:`Memory.context`, counted against the budget like any other."""

_DOCUMENT = "document"
_TURN = "turn"

_SQL_CHUNK = 400  # keeps ``pk IN (...)`` under every SQLite variable limit

_COLUMNS = "pk, id, kind, text, metadata, source, timestamp"


class Memory:
    """A persistent, searchable memory of documents and conversation turns.

    Parameters
    ----------
    path:
        Where to keep the SQLite file.  It is created on demand, parent folders
        included, and reopening the same path finds everything added before.
        ``None`` (the default) gives a private in-memory store that disappears
        on :meth:`close` - handy for tests, useless for persistence.
    namespace:
        An independent compartment inside the one file.  Two namespaces never
        see each other's memories, so one database can hold one namespace per
        user, per project or per tenant.
    embed:
        Optional ``embed(list_of_texts) -> list_of_vectors``.  Given one, every
        memory is stored with its vector and :meth:`search` blends cosine
        similarity into the BM25 ranking with weight :data:`VECTOR_WEIGHT`.
        Without one, search is pure BM25 and no vectors are stored.

    Notes
    -----
    The file is opened in WAL mode with a :data:`BUSY_TIMEOUT` second busy
    timeout, so several processes can read and write the same store without
    raising ``database is locked``.  Within one process the object is also
    safe to share between threads.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        namespace: str = "default",
        embed: Optional[Callable[[List[str]], Sequence]] = None,
    ) -> None:
        if namespace is None or not str(namespace).strip():
            raise ValueError("namespace must be a non-empty string")
        if embed is not None and not callable(embed):
            raise ValueError(
                "embed must be a callable taking a list of texts and returning "
                f"one vector per text, got {type(embed).__name__}"
            )
        self.path = None if path is None else str(path)
        self.namespace = str(namespace)
        self.embed = embed
        self.vector_weight = VECTOR_WEIGHT
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = _store.connect(
            self.path, timeout=BUSY_TIMEOUT
        )
        self._stats_cache: Optional[Tuple[int, int, float]] = None
        self._vector_cache: Optional[Tuple[int, List[int], Any]] = None

    # ------------------------------------------------------------------ core

    def add(
        self,
        text: str,
        *,
        id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        source: Optional[str] = None,
        timestamp: Optional[Any] = None,
    ) -> str:
        """Store one memory and return its id.

        ``id`` is generated when omitted.  Passing an id that already exists
        replaces that memory in place, which makes ``add`` safe to re-run over
        the same documents without piling up duplicates.  ``timestamp`` accepts
        a ``datetime``, a ``date``, an ISO 8601 string or epoch seconds, and
        defaults to now.
        """
        return self.add_many([
            {
                "text": text,
                "id": id,
                "metadata": metadata,
                "source": source,
                "timestamp": timestamp,
            }
        ])[0]

    def add_many(self, items: Iterable[Any]) -> List[str]:
        """Store many memories in one transaction and return their ids.

        Each item may be a plain string, a mapping with a ``text`` key (plus any
        of ``id``, ``metadata``, ``source``, ``timestamp``), or a
        :class:`~document_memory.Record`.  When ``embed`` is set it is called
        once for the whole batch, not once per item.
        """
        prepared = [self._prepare(item, position=n) for n, item in enumerate(items)]
        if not prepared:
            return []
        vectors = self._embed_batch([row["text"] for row in prepared])
        with self._write() as connection:
            for index, row in enumerate(prepared):
                blob = None if vectors is None else _vectors.to_blob(vectors[index])
                self._upsert(connection, row, blob)
        return [row["id"] for row in prepared]

    def get(self, id: str) -> Optional[Record]:
        """The memory with this id, or ``None`` if the store has never held it."""
        row = self._read(
            f"SELECT {_COLUMNS} FROM memories WHERE namespace = ? AND id = ?",
            (self.namespace, str(id)),
        ).fetchone()
        return None if row is None else _record(row)

    def delete(self, id: str) -> bool:
        """Remove one memory.  Returns ``False`` for an id that is not there.

        Deleting something that does not exist is a no-op, not an error: it is
        the normal outcome of two workers cleaning up the same document.
        """
        with self._write() as connection:
            row = connection.execute(
                "SELECT pk FROM memories WHERE namespace = ? AND id = ?",
                (self.namespace, str(id)),
            ).fetchone()
            if row is None:
                return False
            self._drop(connection, [row[0]])
            return True

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Record]:
        """Change a memory's text, its metadata, or both.

        Returns the memory as it now stands, or ``None`` if the id is unknown.
        ``metadata`` is merged key by key into what is already there; pass a key
        with value ``None`` to drop it.  Re-embedding happens automatically when
        the text changes and the store has an ``embed``.
        """
        if text is None and metadata is None:
            return self.get(id)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}")

        vector_blob = None
        if text is not None and self.embed is not None:
            matrix = self._embed_batch([str(text)])
            if matrix is not None:
                vector_blob = _vectors.to_blob(matrix[0])

        with self._write() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE namespace = ? AND id = ?",
                (self.namespace, str(id)),
            ).fetchone()
            if row is None:
                return None
            pk = row["pk"]
            new_text = row["text"] if text is None else str(text)
            merged = _store.load_metadata(row["metadata"])
            if metadata is not None:
                for key, value in metadata.items():
                    if value is None:
                        merged.pop(key, None)
                    else:
                        merged[key] = value
            payload = _store.dump_metadata(merged)

            if text is None:
                connection.execute(
                    "UPDATE memories SET metadata = ? WHERE pk = ?", (payload, pk)
                )
            else:
                tokens = _text.tokenize(new_text)
                sets = "text = ?, metadata = ?, length = ?"
                params: List[Any] = [new_text, payload, len(tokens)]
                if vector_blob is not None:
                    sets += ", vector = ?"
                    params.append(vector_blob)
                params.append(pk)
                connection.execute(f"UPDATE memories SET {sets} WHERE pk = ?", params)
                connection.execute("DELETE FROM postings WHERE pk = ?", (pk,))
                self._index(connection, pk, tokens)

            fresh = connection.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE pk = ?", (pk,)
            ).fetchone()
        return _record(fresh)

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: Optional[Mapping[str, Any]] = None,
        since: Optional[Any] = None,
    ) -> Hits:
        """The ``k`` memories that best match ``query``, best first.

        Scoring is Okapi BM25 (``k1=1.5``, ``b=0.75``).  With ``embed=`` set on
        the store, cosine similarity is blended in with weight
        :data:`VECTOR_WEIGHT`.  ``Hit.score`` is normalised so the best match
        for a query is 1.0; ``Hit.bm25`` is the raw, comparable value and
        ``Hit.why()`` spells out how the number was reached.

        ``where`` filters on metadata - ``{"project": "solar"}``, a list for
        "any of", ``{"user.name": "ada"}`` for a nested key, or a callable for
        anything else.  The keys ``source`` and ``kind`` match those fields
        rather than metadata.  ``since`` keeps only memories at or after a
        ``datetime``, ``date``, ISO string or epoch.

        An empty store, an empty query and a query no memory matches all return
        an empty list rather than raising.
        """
        k = max(0, int(k))
        mode = "BM25 + cosine blend" if self.embed is not None else "lexical BM25"
        empty = Hits([], query=query, mode=mode, k=k, namespace=self.namespace)
        if k == 0 or not query or not str(query).strip():
            return empty

        n_docs, avg_length = self._stats()
        if n_docs == 0:
            return empty

        lexical = self._bm25(str(query), n_docs, avg_length)
        similarity = self._similarity(str(query))
        candidates = set(lexical)
        if similarity:
            candidates.update(
                pk for pk, value in similarity.items() if value >= VECTOR_FLOOR
            )
        if not candidates:
            return empty

        allowed = self._allowed(where, since)
        if allowed is not None:
            candidates &= allowed
            if not candidates:
                return empty

        best_bm25 = max((lexical.get(pk, (0.0, ()))[0] for pk in candidates), default=0.0)
        scored: List[Tuple[float, int]] = []
        for pk in candidates:
            raw, _terms = lexical.get(pk, (0.0, ()))
            normalised = raw / best_bm25 if best_bm25 > 0.0 else 0.0
            if similarity:
                score, _lex, _vec = _vectors.blend(
                    normalised, similarity.get(pk, 0.0), self.vector_weight
                )
            else:
                score = normalised
            if score > 0.0:
                scored.append((score, pk))
        if not scored:
            return empty

        scored.sort(key=lambda pair: (-pair[0], -pair[1]))
        chosen = scored[:k]
        rows = self._rows_by_pk([pk for _score, pk in chosen])

        hits: List[Hit] = []
        for score, pk in chosen:
            row = rows.get(pk)
            if row is None:  # pragma: no cover - only if deleted mid-search
                continue
            raw, terms = lexical.get(pk, (0.0, ()))
            hits.append(
                Hit(
                    id=row["id"],
                    text=row["text"],
                    score=round(float(score), 6),
                    metadata=_store.load_metadata(row["metadata"]),
                    source=row["source"],
                    timestamp=float(row["timestamp"]),
                    kind=row["kind"],
                    bm25=round(float(raw), 6),
                    similarity=(
                        round(float(similarity.get(pk, 0.0)), 6) if similarity else None
                    ),
                    terms=tuple(terms),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, -hit.timestamp))
        return Hits(
            hits,
            query=query,
            mode=mode,
            k=k,
            namespace=self.namespace,
            searched=n_docs,
            vector_weight=self.vector_weight if self.embed is not None else None,
        )

    # ------------------------------------------------------------- listings

    def recent(self, n: int = 10, *, source: Optional[str] = None) -> Records:
        """The ``n`` most recent memories, newest first, optionally one source."""
        n = max(0, int(n))
        if n == 0:
            return Records([], namespace=self.namespace, source=source)
        sql = f"SELECT {_COLUMNS} FROM memories WHERE namespace = ?"
        params: List[Any] = [self.namespace]
        if source is not None:
            sql += " AND source = ?"
            params.append(str(source))
        sql += " ORDER BY timestamp DESC, pk DESC LIMIT ?"
        params.append(n)
        rows = self._read(sql, params).fetchall()
        return Records(
            [_record(row) for row in rows], namespace=self.namespace, source=source
        )

    def count(self) -> int:
        """How many memories this namespace holds."""
        return self._stats()[0]

    def clear(self, namespace: Optional[str] = None) -> int:
        """Delete every memory in a namespace and return how many went.

        Defaults to this object's own namespace.  Pass ``"*"`` to empty the
        whole file, every namespace in it.
        """
        target = self.namespace if namespace is None else str(namespace)
        with self._write() as connection:
            if target == "*":
                removed = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                connection.execute("DELETE FROM postings")
                connection.execute("DELETE FROM memories")
            else:
                removed = connection.execute(
                    "SELECT COUNT(*) FROM memories WHERE namespace = ?", (target,)
                ).fetchone()[0]
                connection.execute("DELETE FROM postings WHERE namespace = ?", (target,))
                connection.execute("DELETE FROM memories WHERE namespace = ?", (target,))
        return int(removed)

    # -------------------------------------------------------- conversations

    def remember(self, role: str, content: str, **meta: Any) -> str:
        """Store one conversation turn and return its id.

        ``role`` ("user", "assistant", "system", a tool name - anything) is kept
        in the metadata, so ``search(..., where={"role": "user"})`` works and
        turns rank alongside documents in ordinary search.  Extra keyword
        arguments join the metadata; ``source`` and ``timestamp`` are lifted out
        and used as such.
        """
        if not role or not str(role).strip():
            raise ValueError("role must be a non-empty string, for example 'user'")
        extra = dict(meta)
        source = extra.pop("source", None)
        timestamp = extra.pop("timestamp", None)
        extra.pop("role", None)
        metadata = {"role": str(role)}
        metadata.update(extra)
        return self.add_many([
            {
                "text": content,
                "metadata": metadata,
                "source": source,
                "timestamp": timestamp,
                "kind": _TURN,
            }
        ])[0]

    def history(self, n: int = 20) -> Turns:
        """The last ``n`` conversation turns, oldest first, ready to replay."""
        n = max(0, int(n))
        if n == 0:
            return Turns([], namespace=self.namespace)
        rows = self._read(
            f"SELECT {_COLUMNS} FROM memories WHERE namespace = ? AND kind = ?"
            " ORDER BY timestamp DESC, pk DESC LIMIT ?",
            (self.namespace, _TURN, n),
        ).fetchall()
        turns = []
        for row in reversed(rows):
            metadata = _store.load_metadata(row["metadata"])
            turns.append(
                Turn(
                    id=row["id"],
                    role=str(metadata.get("role", "user")),
                    content=row["text"],
                    timestamp=float(row["timestamp"]),
                    metadata=metadata,
                    source=row["source"],
                )
            )
        return Turns(turns, namespace=self.namespace)

    def context(
        self,
        query: str,
        *,
        budget: int = 2000,
        counter: Optional[Callable[[str], int]] = None,
    ) -> str:
        """The best-matching memories, packed to ``budget`` and ready to paste.

        Memories are taken in relevance order, ties broken newest-first, and
        added one at a time for as long as the *whole* returned string stays
        within the budget - the header and the blank lines are counted too, so
        ``counter(memory.context(q, budget=b)) <= b`` always holds.

        ``counter`` measures the budget and defaults to whitespace-separated
        words.  Pass your model's tokeniser (for example
        ``counter=lambda s: len(enc.encode(s))``) to budget in real tokens.
        A budget too small for even one memory returns an empty string.
        """
        budget = int(budget)
        measure = counter if counter is not None else _words
        if not callable(measure):
            raise ValueError("counter must be a callable taking a string and returning an int")
        if budget <= 0:
            return ""

        hits = self.search(query, k=_CONTEXT_POOL)
        if not hits:
            return ""
        ordered = sorted(hits, key=lambda hit: (-hit.score, -hit.timestamp))

        header = CONTEXT_HEADER.format(query=str(query))
        if int(measure(header)) > budget:
            return ""

        blocks: List[str] = []
        for hit in ordered:
            block = _block(hit)
            candidate = _join(header, blocks + [block])
            if int(measure(candidate)) <= budget:
                blocks.append(block)
        if not blocks:
            trimmed = _shrink(ordered[0], header, budget, measure)
            if trimmed is None:
                return ""
            blocks.append(trimmed)
        return _join(header, blocks)

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the database handle.  Calling it twice is fine."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._stats_cache = None
            self._vector_cache = None

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        where = self.path or ":memory:"
        mode = "bm25+vectors" if self.embed is not None else "bm25"
        return f"Memory({where!r}, namespace={self.namespace!r}, {mode})"

    def summary(self) -> str:
        """One human-readable line describing the store."""
        where = self.path or "in-memory (not persisted)"
        mode = (
            f"BM25 + cosine, vector weight {self.vector_weight:g}"
            if self.embed is not None
            else "BM25"
        )
        total = self.count()
        turns = self._read(
            "SELECT COUNT(*) FROM memories WHERE namespace = ? AND kind = ?",
            (self.namespace, _TURN),
        ).fetchone()[0]
        noun = "memory" if total == 1 else "memories"
        return (
            f"{where}  namespace {self.namespace!r}  {total} {noun} "
            f"({int(turns)} conversation turns)  ranking: {mode}"
        )

    # -------------------------------------------------------------- private

    def _require_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ValueError(
                "this Memory is closed; open a new one with Memory(path) "
                "(the data is still on disk)"
            )
        return self._connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside ``BEGIN IMMEDIATE``, bumping the revision counter."""
        with self._lock:
            connection = self._require_open()
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            _store.bump_revision(connection)
            connection.execute("COMMIT")
            self._stats_cache = None
            self._vector_cache = None

    def _read(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._require_open().execute(sql, tuple(params))

    def _prepare(self, item: Any, *, position: int) -> Dict[str, Any]:
        """Normalise one ``add_many`` item into the dict the writer expects."""
        if isinstance(item, Record):
            item = {
                "text": item.text,
                "id": item.id,
                "metadata": item.metadata,
                "source": item.source,
                "timestamp": item.timestamp,
                "kind": item.kind,
            }
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, Mapping):
            raise ValueError(
                f"item {position} must be a string, a dict with a 'text' key or a "
                f"Record, got {type(item).__name__}"
            )
        if "text" not in item:
            raise ValueError(
                f"item {position} has no 'text' key (it has: "
                f"{', '.join(sorted(map(str, item))) or 'nothing'})"
            )
        text = item["text"]
        if text is None:
            raise ValueError(f"item {position} has text=None; store a string")
        if not isinstance(text, str):
            raise ValueError(
                f"item {position} has text of type {type(text).__name__}; store a string"
            )
        identifier = item.get("id")
        identifier = uuid.uuid4().hex if identifier is None else str(identifier)
        if not identifier.strip():
            raise ValueError(f"item {position} has an empty id; omit it to get one")
        source = item.get("source")
        metadata = item.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError(
                f"item {position} metadata must be a dict, got {type(metadata).__name__}"
            )
        return {
            "id": identifier,
            "text": text,
            "metadata": _store.dump_metadata(
                dict(metadata) if metadata is not None else None,
                field=f"item {position} metadata",
            ),
            "source": None if source is None else str(source),
            "timestamp": _store.to_epoch(
                item.get("timestamp"), field=f"item {position} timestamp"
            ),
            "kind": str(item.get("kind") or _DOCUMENT),
            "tokens": _text.tokenize(text),
        }

    def _embed_batch(self, texts: Sequence[str]) -> Optional[Any]:
        if self.embed is None:
            return None
        return _vectors.embed_texts(self.embed, texts, expected_dim=self._vector_dim())

    def _vector_dim(self) -> Optional[int]:
        row = self._read(
            "SELECT LENGTH(vector) FROM memories WHERE namespace = ? AND vector IS NOT NULL"
            " LIMIT 1",
            (self.namespace,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        return int(row[0]) // _vectors.DTYPE.itemsize

    def _upsert(
        self, connection: sqlite3.Connection, row: Dict[str, Any], blob: Optional[bytes]
    ) -> None:
        existing = connection.execute(
            "SELECT pk FROM memories WHERE namespace = ? AND id = ?",
            (self.namespace, row["id"]),
        ).fetchone()
        tokens = row["tokens"]
        if existing is not None:
            pk = existing[0]
            connection.execute(
                "UPDATE memories SET kind = ?, text = ?, metadata = ?, source = ?,"
                " timestamp = ?, length = ?, vector = ? WHERE pk = ?",
                (
                    row["kind"],
                    row["text"],
                    row["metadata"],
                    row["source"],
                    row["timestamp"],
                    len(tokens),
                    blob,
                    pk,
                ),
            )
            connection.execute("DELETE FROM postings WHERE pk = ?", (pk,))
        else:
            cursor = connection.execute(
                "INSERT INTO memories(namespace, id, kind, text, metadata, source,"
                " timestamp, created, length, vector)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.namespace,
                    row["id"],
                    row["kind"],
                    row["text"],
                    row["metadata"],
                    row["source"],
                    row["timestamp"],
                    row["timestamp"],
                    len(tokens),
                    blob,
                ),
            )
            pk = int(cursor.lastrowid)
        self._index(connection, pk, tokens)

    def _index(self, connection: sqlite3.Connection, pk: int, tokens: Sequence[str]) -> None:
        counts = _text.term_frequencies(tokens)
        if not counts:
            return
        length = len(tokens)
        connection.executemany(
            "INSERT OR REPLACE INTO postings(namespace, term, pk, tf, length)"
            " VALUES (?, ?, ?, ?, ?)",
            [(self.namespace, term, pk, tf, length) for term, tf in counts.items()],
        )

    def _drop(self, connection: sqlite3.Connection, pks: Sequence[int]) -> None:
        for chunk in _chunks(list(pks), _SQL_CHUNK):
            marks = ",".join("?" * len(chunk))
            connection.execute(f"DELETE FROM postings WHERE pk IN ({marks})", chunk)
            connection.execute(f"DELETE FROM memories WHERE pk IN ({marks})", chunk)

    def _stats(self) -> Tuple[int, float]:
        """``(document count, average token length)`` for this namespace."""
        with self._lock:
            current = _store.revision(self._require_open())
            if self._stats_cache is not None and self._stats_cache[0] == current:
                return self._stats_cache[1], self._stats_cache[2]
            row = self._read(
                "SELECT COUNT(*), COALESCE(AVG(length), 0.0) FROM memories"
                " WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
            n_docs, avg_length = int(row[0]), float(row[1])
            self._stats_cache = (current, n_docs, avg_length)
            return n_docs, avg_length

    def _bm25(
        self, query: str, n_docs: int, avg_length: float
    ) -> Dict[int, Tuple[float, Tuple[str, ...]]]:
        """BM25 score and matched terms per candidate pk."""
        terms = sorted(set(_text.tokenize(query)))
        if not terms:
            return {}
        scores: Dict[int, float] = {}
        matched: Dict[int, List[str]] = {}
        for term in terms:
            rows = self._read(
                "SELECT pk, tf, length FROM postings WHERE namespace = ? AND term = ?",
                (self.namespace, term),
            ).fetchall()
            if not rows:
                continue
            weight = _text.idf(n_docs, len(rows))
            for pk, tf, length in rows:
                scores[pk] = scores.get(pk, 0.0) + _text.bm25_term_score(
                    tf, length, avg_length, weight
                )
                matched.setdefault(pk, []).append(term)
        return {pk: (score, tuple(matched.get(pk, ()))) for pk, score in scores.items()}

    def _similarity(self, query: str) -> Dict[int, float]:
        """Cosine of the query against every stored vector, or ``{}`` if lexical."""
        if self.embed is None:
            return {}
        pks, matrix = self._vectors_for_namespace()
        if not pks:
            return {}
        query_matrix = _vectors.embed_texts(
            self.embed, [query], expected_dim=matrix.shape[1]
        )
        values = _vectors.cosine(query_matrix[0], matrix)
        return {pk: float(value) for pk, value in zip(pks, values)}

    def _vectors_for_namespace(self) -> Tuple[List[int], Any]:
        import numpy as np

        with self._lock:
            current = _store.revision(self._require_open())
            if self._vector_cache is not None and self._vector_cache[0] == current:
                return self._vector_cache[1], self._vector_cache[2]
            rows = self._read(
                "SELECT pk, vector FROM memories WHERE namespace = ? AND vector IS NOT NULL"
                " ORDER BY pk",
                (self.namespace,),
            ).fetchall()
            pks = [int(row[0]) for row in rows]
            dim = self._vector_dim() or 1
            matrix = _vectors.from_blobs([row[1] for row in rows], dim)
            if not pks:
                matrix = np.zeros((0, dim), dtype="float32")
            self._vector_cache = (current, pks, matrix)
            return pks, matrix

    def _allowed(
        self, where: Optional[Mapping[str, Any]], since: Optional[Any]
    ) -> Optional[set]:
        """The pks passing ``where`` / ``since``, or ``None`` when both are absent."""
        if where is None and since is None:
            return None
        if where is not None and not isinstance(where, Mapping):
            raise ValueError(
                "where must be a dict of metadata key -> value, list of values or "
                f"predicate, got {type(where).__name__}"
            )
        sql = "SELECT pk, kind, source, metadata FROM memories WHERE namespace = ?"
        params: List[Any] = [self.namespace]
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(_store.to_epoch(since, field="since"))
        rows = self._read(sql, params).fetchall()
        if not where:
            return {int(row[0]) for row in rows}
        allowed = set()
        for pk, kind, source, raw in rows:
            fields = {"kind": kind, "source": source}
            metadata = _store.load_metadata(raw)
            if all(
                _matches(key, want, fields, metadata) for key, want in where.items()
            ):
                allowed.add(int(pk))
        return allowed

    def _rows_by_pk(self, pks: Sequence[int]) -> Dict[int, sqlite3.Row]:
        found: Dict[int, sqlite3.Row] = {}
        for chunk in _chunks(list(pks), _SQL_CHUNK):
            marks = ",".join("?" * len(chunk))
            for row in self._read(
                f"SELECT {_COLUMNS} FROM memories WHERE pk IN ({marks})", chunk
            ).fetchall():
                found[int(row["pk"])] = row
        return found


# --------------------------------------------------------------- module helpers

_CONTEXT_POOL = 50
"""How many ranked memories ``context()`` may choose from while packing."""


def _record(row: sqlite3.Row) -> Record:
    return Record(
        id=row["id"],
        text=row["text"],
        metadata=_store.load_metadata(row["metadata"]),
        source=row["source"],
        timestamp=float(row["timestamp"]),
        kind=row["kind"],
    )


def _chunks(values: List[Any], size: int) -> Iterator[List[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _lookup(data: Mapping[str, Any], key: str) -> Tuple[Any, bool]:
    """Fetch ``key`` from metadata, falling back to a dotted path into nested dicts."""
    if key in data:
        return data[key], True
    if "." in key:
        current: Any = data
        for part in key.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None, False
        return current, True
    return None, False


def _matches(
    key: str, want: Any, fields: Mapping[str, Any], metadata: Mapping[str, Any]
) -> bool:
    """Does one ``where`` clause hold for one row?"""
    if key in fields and key not in metadata:
        value, found = fields[key], True
    else:
        value, found = _lookup(metadata, key)
    if callable(want):
        return bool(want(value if found else None))
    if not found:
        return False
    if isinstance(want, (list, tuple, set, frozenset)):
        return any(value == option for option in want)
    return value == want


def _words(text: str) -> int:
    """Default ``context`` budget unit: whitespace-separated words."""
    return len(text.split())


def _block(hit: Hit) -> str:
    """One memory as it appears inside ``context()``."""
    label = hit.source or hit.metadata.get("role") or hit.kind
    body = " ".join(str(hit.text).split())
    return f"[{_store.day(hit.timestamp)} | {label}] {body}"


def _join(header: str, blocks: Sequence[str]) -> str:
    return header + "\n\n" + "\n\n".join(blocks)


def _shrink(
    hit: Hit, header: str, budget: int, measure: Callable[[str], int]
) -> Optional[str]:
    """The longest word-prefix of ``hit`` that still fits under ``budget``."""
    block = _block(hit)
    words = block.split()
    low, high, best = 0, len(words), None
    while low <= high:
        middle = (low + high) // 2
        candidate = " ".join(words[:middle]) + (" ..." if middle < len(words) else "")
        if middle and int(measure(_join(header, [candidate]))) <= budget:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best
