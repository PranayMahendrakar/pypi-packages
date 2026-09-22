"""Two connections on one file must not raise "database is locked".

SQLite's defaults are the problem this file guards: the rollback journal makes a
writer block every reader, and a busy timeout of 0 makes the loser of a race
fail immediately with ``sqlite3.OperationalError: database is locked`` instead of
waiting its turn.  :func:`document_memory._store.connect` turns on WAL
journalling and a :data:`~document_memory.BUSY_TIMEOUT` second busy timeout, and
every write runs inside ``BEGIN IMMEDIATE`` so two connections cannot deadlock
while upgrading a read transaction they already hold.

The tests below open genuinely separate connections - separate ``Memory``
objects, and in one case separate processes - and write from all of them.
"""
from __future__ import annotations

import subprocess
import sqlite3
import sys
import threading

import pytest

from document_memory import BUSY_TIMEOUT, Memory


# ------------------------------------------------------- the pragmas are really on


def test_a_file_store_is_opened_in_wal_mode_with_a_busy_timeout(tmp_path):
    """WAL and the busy timeout are what make everything else in this file work."""
    path = tmp_path / "notes.db"
    with Memory(path) as memory:
        connection = memory._connection
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == int(
            BUSY_TIMEOUT * 1000
        )


def test_the_busy_timeout_is_long_enough_to_be_worth_having():
    """A timeout of zero would be the same as having none at all."""
    assert BUSY_TIMEOUT >= 5.0


def test_wal_mode_survives_a_reopen(tmp_path):
    """WAL is a property of the file, so the second opener inherits it."""
    path = tmp_path / "notes.db"
    with Memory(path) as first:
        first.add("solar panels turn light into power")
    with Memory(path) as second:
        mode = second._connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


# ------------------------------------------------- two connections, both writing


def test_two_connections_write_to_one_store_without_locking(tmp_path):
    """The headline case: two open connections, both writing, interleaved."""
    path = tmp_path / "notes.db"
    with Memory(path) as left, Memory(path) as right:
        for n in range(25):
            left.add(f"left memory {n} about solar panels", id=f"left-{n}")
            right.add(f"right memory {n} about wind turbines", id=f"right-{n}")

        # Each connection sees its own writes and the other's.
        assert left.count() == 50
        assert right.count() == 50
        assert left.get("right-7") is not None
        assert right.get("left-7") is not None

        # And the index is shared too, not just the rows.
        assert {hit.id for hit in left.search("turbines", k=50)} == {
            f"right-{n}" for n in range(25)
        }


def test_a_reader_is_not_blocked_by_an_open_writer(tmp_path):
    """WAL's real point: a search in one connection works mid-write in another."""
    path = tmp_path / "notes.db"
    with Memory(path) as writer, Memory(path) as reader:
        writer.add("solar panels turn light into power", id="seed")
        with writer._write() as connection:
            connection.execute(
                "INSERT INTO memories(namespace, id, kind, text, metadata, source,"
                " timestamp, created, length, vector)"
                " VALUES ('default', 'held', 'document', 'held open', '{}', NULL,"
                " 0.0, 0.0, 2, NULL)"
            )
            # The write transaction above is still open right here.
            assert reader.search("solar")[0].id == "seed"
            assert reader.count() == 1
        assert reader.count() == 2


def test_two_threads_hammering_the_same_file_never_see_database_is_locked(tmp_path):
    """Separate connections, released together, writing as fast as they can."""
    path = tmp_path / "notes.db"
    start = threading.Barrier(2)
    failures = []

    def writer(tag: str) -> None:
        try:
            with Memory(path) as memory:
                start.wait(timeout=10)
                for n in range(40):
                    memory.add(f"{tag} memory {n} about electricity", id=f"{tag}-{n}")
        except Exception as exc:  # pragma: no cover - the failure this test guards
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(tag,)) for tag in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "a writer thread never finished"

    assert failures == [], failures
    with Memory(path) as memory:
        assert memory.count() == 80
        assert len(memory.search("electricity", k=100)) == 80


def test_one_object_shared_between_threads_is_also_safe(tmp_path):
    """Within a process the lock in Memory serialises writes on one connection."""
    path = tmp_path / "notes.db"
    failures = []

    with Memory(path) as memory:
        def writer(tag: str) -> None:
            try:
                for n in range(30):
                    memory.add(f"{tag} shared {n}", id=f"{tag}-{n}")
                    memory.search("shared", k=3)
            except Exception as exc:  # pragma: no cover - the failure this guards
                failures.append(f"{tag}: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=writer, args=(tag,)) for tag in "xyz"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive()

        assert failures == [], failures
        assert memory.count() == 90


# --------------------------------------------------- separate processes entirely

_CHILD = """
import sys
from document_memory import Memory

path, tag = sys.argv[1], sys.argv[2]
with Memory(path) as memory:
    for n in range(30):
        memory.add(tag + " memory " + str(n) + " about electricity", id=tag + "-" + str(n))
print(memory.path is not None)
"""


def test_two_processes_write_to_one_store_without_locking(tmp_path):
    """The case WAL and the busy timeout exist for: no shared Python lock at all."""
    path = str(tmp_path / "notes.db")
    with Memory(path) as memory:
        memory.add("seed", id="seed")

    children = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, path, tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for tag in ("p1", "p2")
    ]
    results = []
    for child in children:
        out, err = child.communicate(timeout=120)
        results.append((child.returncode, out, err))

    for code, _out, err in results:
        assert "database is locked" not in err
        assert code == 0, err

    with Memory(path) as memory:
        assert memory.count() == 61
        assert len(memory.search("electricity", k=100)) == 60


# ------------------------------------------------------------- persistence again

def test_a_write_from_one_connection_is_durable_for_the_next_opener(tmp_path):
    """Closing and reopening across connections loses nothing."""
    path = tmp_path / "notes.db"
    with Memory(path) as first:
        first.add("solar panels turn light into power", id="solar")
    with Memory(path) as second:
        second.add("wind turbines turn air into power", id="wind")
    with Memory(path) as third:
        assert third.count() == 2
        assert third.get("solar").text.startswith("solar panels")
        assert third.get("wind").text.startswith("wind turbines")


def test_a_path_that_is_not_a_store_says_so_rather_than_crashing(tmp_path):
    """Pointing Memory at an unrelated file is a clear ValueError, not a traceback."""
    path = tmp_path / "not-a-database.db"
    path.write_bytes(b"this is definitely not a SQLite file" * 20)
    with pytest.raises(ValueError, match="not a document-memory store"):
        Memory(path)


def test_namespaces_do_not_collide_across_connections(tmp_path):
    """Two connections, two namespaces, one file: neither sees the other."""
    path = tmp_path / "notes.db"
    with Memory(path, namespace="ada") as ada, Memory(path, namespace="alan") as alan:
        ada.add("ada's note about solar panels", id="n1")
        alan.add("alan's note about solar panels", id="n1")
        assert ada.count() == 1
        assert alan.count() == 1
        assert ada.get("n1").text.startswith("ada's")
        assert alan.get("n1").text.startswith("alan's")
        assert ada.search("solar")[0].text.startswith("ada's")


def test_sqlite_itself_still_reports_a_lock_without_our_settings(tmp_path):
    """Control case: the default connection is the one that fails, ours does not.

    This is what proves the pragmas are load-bearing rather than decorative.
    """
    path = str(tmp_path / "notes.db")
    with Memory(path) as memory:
        memory.add("seed", id="seed")
        blocker = sqlite3.connect(path, timeout=0)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            blocker.execute(
                "INSERT INTO memories(namespace, id, kind, text, metadata, source,"
                " timestamp, created, length, vector)"
                " VALUES ('default', 'blocked', 'document', 'x', '{}', NULL,"
                " 0.0, 0.0, 1, NULL)"
            )
            # A second no-timeout connection gives up at once...
            impatient = sqlite3.connect(path, timeout=0)
            try:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    impatient.execute("BEGIN IMMEDIATE")
            finally:
                impatient.close()
            # ...while a reader through Memory carries on regardless.
            assert memory.search("seed")[0].id == "seed"
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        memory.add("after the lock cleared", id="after")
        assert memory.count() == 2
