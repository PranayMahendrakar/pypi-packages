"""The SQLite layer: one table of entries, safe for several processes at once.

WAL journalling plus a busy timeout is what lets two processes write to the
same cache file without either of them seeing "database is locked". A file
that is not a database at all (truncated, half-written, or something else
entirely) is moved aside and recreated rather than raised at the caller.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("prompt_cache")

#: Layout version of the ``entries`` table, recorded in ``meta``.
SCHEMA_VERSION = "1"

MEMORY = ":memory:"

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS entries (
        key       TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        fmt       TEXT NOT NULL,
        value     BLOB NOT NULL,
        size      INTEGER NOT NULL,
        created   REAL NOT NULL,
        accessed  REAL NOT NULL,
        expires   REAL,
        hits      INTEGER NOT NULL DEFAULT 0,
        preview   TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS entries_namespace_accessed ON entries (namespace, accessed)",
    "CREATE INDEX IF NOT EXISTS entries_expires ON entries (expires)",
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)

_CORRUPTION_MARKERS = (
    "not a database",
    "malformed",
    "corrupt",
    "encrypted",
    "database disk image",
)


def is_corruption(exc: BaseException) -> bool:
    """True when a sqlite3 error means the file is unusable, not merely busy."""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _CORRUPTION_MARKERS)


class Store:
    """A thin, thread-safe wrapper around the cache database.

    The connection is opened the first time it is needed, so building a
    :class:`~prompt_cache.cache.Cache` never touches the filesystem.
    """

    def __init__(self, path: Union[str, Path], *, timeout: float = 15.0) -> None:
        self._memory = str(path) == MEMORY
        self._path = MEMORY if self._memory else Path(path)
        self._timeout = float(timeout)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self.rebuilt = False

    # ---------------------------------------------------------------- connect

    @property
    def path(self) -> Union[str, Path]:
        """The database file (or ``":memory:"``)."""
        return self._path

    def _open(self) -> sqlite3.Connection:
        fresh = True
        if not self._memory:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fresh = not self._path.exists() or self._path.stat().st_size == 0
        conn = sqlite3.connect(
            str(self._path),
            timeout=self._timeout,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute("PRAGMA busy_timeout = {0}".format(int(self._timeout * 1000)))
            if not self._memory:
                if fresh:
                    # Must be set before the first table exists; keeps the file
                    # from growing without bound once entries are evicted.
                    conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                try:
                    conn.execute("PRAGMA journal_mode = WAL")
                except sqlite3.DatabaseError as exc:
                    if is_corruption(exc):
                        raise
                    # e.g. a network share: slower, but still correct.
                    logger.warning("prompt-cache: WAL journalling unavailable at %s", self._path)
                conn.execute("PRAGMA synchronous = NORMAL")
            for statement in _DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            # Probe: a file that is not a database fails here, not in the caller.
            conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        except BaseException:
            # Let go of the file before anything tries to move or delete it.
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover - close rarely fails
                pass
            raise
        return conn

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = self._open()
            except sqlite3.DatabaseError as exc:
                if not is_corruption(exc):
                    raise
                self._rebuild(exc)
                self._conn = self._open()
        return self._conn

    def _rebuild(self, exc: BaseException) -> None:
        """Move a corrupt database aside so a fresh, empty one can take over."""
        self._close_quietly()
        if self._memory:  # pragma: no cover - an in-memory db cannot corrupt
            return
        logger.warning(
            "prompt-cache: %s is not a usable database (%s); starting a new one",
            self._path,
            exc,
        )
        broken = Path(str(self._path) + ".corrupt")
        try:
            os.replace(str(self._path), str(broken))
        except OSError:
            try:
                os.remove(str(self._path))
            except OSError:
                pass
        for suffix in ("-wal", "-shm"):
            try:
                os.remove(str(self._path) + suffix)
            except OSError:
                pass
        self.rebuilt = True

    def _close_quietly(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - close rarely fails
                pass
            self._conn = None

    def close(self) -> None:
        """Release the connection. The store reopens if it is used again."""
        with self._lock:
            self._close_quietly()

    def _run(self, work) -> Any:
        """Run ``work(conn)``, rebuilding once if the file turns out corrupt."""
        with self._lock:
            try:
                return work(self._connection())
            except sqlite3.DatabaseError as exc:
                if not is_corruption(exc):
                    raise
                self._rebuild(exc)
                return work(self._connection())

    # ----------------------------------------------------------------- reads

    def lookup(self, key: str, now: float) -> Optional[Tuple[str, bytes]]:
        """Return ``(format, blob)`` for a live entry, or ``None``.

        An entry found past its expiry is deleted and reported as a miss.
        """

        def work(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT fmt, value, expires FROM entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            fmt, blob, expires = row
            if expires is not None and expires <= now:
                conn.execute("DELETE FROM entries WHERE key = ?", (key,))
                return None
            conn.execute(
                "UPDATE entries SET accessed = ?, hits = hits + 1 WHERE key = ?", (now, key)
            )
            return fmt, bytes(blob)

        return self._run(work)

    def totals(self, namespace: str, now: float) -> Tuple[int, int]:
        """``(entry count, total value bytes)`` for live entries in a namespace."""

        def work(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM entries "
                "WHERE namespace = ? AND (expires IS NULL OR expires > ?)",
                (namespace, now),
            ).fetchone()
            return int(row[0]), int(row[1])

        return self._run(work)

    def entries(self, namespace: str, now: float, limit: int = 20) -> List[Dict[str, Any]]:
        """Most recently used live entries first, as plain dicts."""

        def work(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT key, preview, size, created, accessed, expires, hits, fmt "
                "FROM entries WHERE namespace = ? AND (expires IS NULL OR expires > ?) "
                "ORDER BY accessed DESC LIMIT ?",
                (namespace, now, int(limit)),
            ).fetchall()
            return [
                {
                    "key": row[0],
                    "preview": row[1],
                    "size": int(row[2]),
                    "created": float(row[3]),
                    "accessed": float(row[4]),
                    "expires": None if row[5] is None else float(row[5]),
                    "hits": int(row[6]),
                    "format": row[7],
                }
                for row in rows
            ]

        return self._run(work)

    def namespaces(self) -> List[str]:
        """Every namespace present in the file, sorted."""

        def work(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT namespace FROM entries GROUP BY namespace ORDER BY namespace"
            ).fetchall()
            return [str(row[0]) for row in rows]

        return self._run(work)

    # ---------------------------------------------------------------- writes

    def store(
        self,
        key: str,
        namespace: str,
        fmt: str,
        blob: bytes,
        now: float,
        expires: Optional[float],
        preview: str,
    ) -> None:
        """Insert or replace one entry."""

        def work(conn: sqlite3.Connection):
            conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(key, namespace, fmt, value, size, created, accessed, expires, hits, preview) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (key, namespace, fmt, memoryview(blob), len(blob), now, now, expires, preview),
            )
            return None

        self._run(work)

    def delete(self, key: str) -> int:
        """Remove one entry by key; returns how many rows went."""

        def work(conn: sqlite3.Connection):
            return int(conn.execute("DELETE FROM entries WHERE key = ?", (key,)).rowcount)

        return self._run(work)

    def clear(self, namespace: Optional[str]) -> int:
        """Delete a namespace, or every namespace when ``namespace`` is None."""

        def work(conn: sqlite3.Connection):
            if namespace is None:
                return int(conn.execute("DELETE FROM entries").rowcount)
            return int(
                conn.execute("DELETE FROM entries WHERE namespace = ?", (namespace,)).rowcount
            )

        removed = self._run(work)
        self._reclaim()
        return max(removed, 0)

    def prune(self, namespace: Optional[str], now: float) -> int:
        """Delete entries whose time to live has run out."""

        def work(conn: sqlite3.Connection):
            if namespace is None:
                cur = conn.execute(
                    "DELETE FROM entries WHERE expires IS NOT NULL AND expires <= ?", (now,)
                )
            else:
                cur = conn.execute(
                    "DELETE FROM entries WHERE namespace = ? AND expires IS NOT NULL "
                    "AND expires <= ?",
                    (namespace, now),
                )
            return int(cur.rowcount)

        removed = self._run(work)
        if removed > 0:
            self._reclaim()
        return max(removed, 0)

    def evict(self, namespace: str, max_bytes: int, now: float) -> int:
        """Drop least-recently-used entries until the namespace fits again."""

        def work(conn: sqlite3.Connection):
            removed = 0
            conn.execute("BEGIN IMMEDIATE")
            try:
                removed += int(
                    conn.execute(
                        "DELETE FROM entries WHERE namespace = ? AND expires IS NOT NULL "
                        "AND expires <= ?",
                        (namespace, now),
                    ).rowcount
                )
                row = conn.execute(
                    "SELECT COALESCE(SUM(size), 0) FROM entries WHERE namespace = ?",
                    (namespace,),
                ).fetchone()
                total = int(row[0])
                while total > max_bytes:
                    victims = conn.execute(
                        "SELECT key, size FROM entries WHERE namespace = ? "
                        "ORDER BY accessed ASC, created ASC LIMIT 64",
                        (namespace,),
                    ).fetchall()
                    if not victims:
                        break
                    for key, size in victims:
                        conn.execute("DELETE FROM entries WHERE key = ?", (key,))
                        total -= int(size)
                        removed += 1
                        if total <= max_bytes:
                            break
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return removed

        removed = self._run(work)
        if removed > 0:
            self._reclaim()
        return max(removed, 0)

    def _reclaim(self) -> None:
        """Hand freed pages back to the filesystem (no-op without auto_vacuum)."""

        def work(conn: sqlite3.Connection):
            try:
                conn.execute("PRAGMA incremental_vacuum")
            except sqlite3.DatabaseError:  # pragma: no cover - depends on auto_vacuum
                pass
            return None

        try:
            self._run(work)
        except sqlite3.Error:  # pragma: no cover - housekeeping must never raise
            pass


__all__ = ["MEMORY", "SCHEMA_VERSION", "Store", "is_corruption", "logger"]
