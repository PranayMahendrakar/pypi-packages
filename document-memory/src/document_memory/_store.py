"""SQLite connection, schema and the small coercions the API needs.

The store is one SQLite file (or an in-memory database when no path is given).
Two settings make it safe for several processes at once:

* **WAL journalling** - readers never block the writer and the writer never
  blocks readers, so a search in one process cannot make an ``add`` in another
  fail.
* **A busy timeout** - when two processes do try to write at the same instant,
  the second waits for the lock instead of raising
  ``sqlite3.OperationalError: database is locked``.

Every write runs inside ``BEGIN IMMEDIATE``.  Taking the write lock up front is
what stops two connections from deadlocking while trying to upgrade a read
transaction they already hold.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import date, datetime, timezone
from typing import Any, Optional

MEMORY_PATH = ":memory:"

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS memories (
        pk         INTEGER PRIMARY KEY,
        namespace  TEXT    NOT NULL,
        id         TEXT    NOT NULL,
        kind       TEXT    NOT NULL,
        text       TEXT    NOT NULL,
        metadata   TEXT    NOT NULL,
        source     TEXT,
        timestamp  REAL    NOT NULL,
        created    REAL    NOT NULL,
        length     INTEGER NOT NULL,
        vector     BLOB
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_key ON memories(namespace, id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_len ON memories(namespace, length)",
    "CREATE INDEX IF NOT EXISTS idx_memories_recent ON memories(namespace, timestamp, pk)",
    "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(namespace, kind, timestamp)",
    """
    CREATE TABLE IF NOT EXISTS postings (
        namespace TEXT    NOT NULL,
        term      TEXT    NOT NULL,
        pk        INTEGER NOT NULL,
        tf        INTEGER NOT NULL,
        length    INTEGER NOT NULL,
        PRIMARY KEY (namespace, term, pk)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS idx_postings_pk ON postings(pk)",
    "CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "INSERT OR IGNORE INTO store_meta(key, value) VALUES ('revision', '0')",
    "INSERT OR IGNORE INTO store_meta(key, value) VALUES ('schema_version', '1')",
)

SCHEMA_VERSION = 1


def connect(path: Optional[str], timeout: float = 10.0) -> sqlite3.Connection:
    """Open (creating on demand) the store at ``path`` and install the schema.

    ``path`` of ``None`` gives a private in-memory database.  Parent folders of
    a real path are created if they do not exist, so ``Memory("var/notes.db")``
    works on a fresh checkout.
    """
    target = MEMORY_PATH if path is None else str(path)
    if target != MEMORY_PATH:
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
    connection = sqlite3.connect(
        target,
        timeout=max(0.0, float(timeout)),
        isolation_level=None,          # explicit transactions, see write()
        check_same_thread=False,       # guarded by a lock in Memory
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout = %d" % int(max(0.0, timeout) * 1000))
        if target != MEMORY_PATH:
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:  # pragma: no cover - exotic filesystems
                pass
            connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        create_schema(connection)
    except sqlite3.DatabaseError as exc:
        connection.close()
        raise ValueError(
            f"{target!r} is not a document-memory store ({exc}); point the path at a "
            "new file, or at one an earlier Memory() created"
        ) from None
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the tables if they are missing, tolerating a concurrent creator."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in SCHEMA:
            connection.execute(statement)
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def bump_revision(connection: sqlite3.Connection) -> None:
    """Advance the counter that tells cached vectors they are stale."""
    connection.execute(
        "UPDATE store_meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
        " WHERE key = 'revision'"
    )


def revision(connection: sqlite3.Connection) -> int:
    """The current write counter; any change invalidates an in-process cache."""
    row = connection.execute(
        "SELECT value FROM store_meta WHERE key = 'revision'"
    ).fetchone()
    return int(row[0]) if row else 0


def to_epoch(value: Any, *, field: str = "timestamp") -> float:
    """Coerce a datetime / date / ISO string / epoch number to epoch seconds.

    Naive datetimes are read as UTC so the same code gives the same answer on
    every machine.
    """
    if value is None:
        return time.time()
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a datetime, an ISO string or a number, not a bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp()
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return to_epoch(datetime.fromisoformat(text), field=field)
        except ValueError:
            raise ValueError(
                f"{field} string {value!r} is not ISO 8601, for example '2026-09-22' "
                "or '2026-09-22T10:30:00'"
            ) from None
    raise ValueError(
        f"{field} must be a datetime, a date, an ISO 8601 string or epoch seconds, "
        f"got {type(value).__name__}"
    )


def iso(epoch: float) -> str:
    """Epoch seconds as a UTC ISO 8601 string, for JSON output and summaries."""
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat(timespec="seconds")


def day(epoch: float) -> str:
    """Epoch seconds as a UTC ``YYYY-MM-DD`` date, for context() labels."""
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%d")


def dump_metadata(metadata: Optional[dict], *, field: str = "metadata") -> str:
    """Serialise metadata, rejecting anything JSON cannot carry with a clear error."""
    if metadata is None:
        return "{}"
    if not isinstance(metadata, dict):
        raise ValueError(f"{field} must be a dict, got {type(metadata).__name__}")
    try:
        return json.dumps(metadata, ensure_ascii=False, sort_keys=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be JSON-serialisable (dicts, lists, strings, numbers, "
            f"bools, None): {exc}"
        ) from None


def load_metadata(raw: Optional[str]) -> dict:
    """Parse stored metadata; a corrupt row degrades to ``{}`` rather than raising."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):  # pragma: no cover - only a hand-edited row
        return {}
    return value if isinstance(value, dict) else {}
