"""Append-only JSONL storage.

One directory per watchdog, one file inside it. Each record is one JSON object
on one line, written under an exclusive lock so several processes appending at
the same time interleave whole lines instead of overwriting each other.

POSIX gets ``O_APPEND`` (atomic for regular files) plus an advisory
``flock``; Windows has no atomic append, so the write is serialised with a
byte-range lock taken far beyond end-of-file - a region no reader ever touches,
which matters because Windows byte-range locks are mandatory and would
otherwise block readers.

Reading skips any line that does not parse: a half-written trailing line from a
process that died mid-write costs a warning, not a crash.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ._stats import jsonable

logger = logging.getLogger(__name__)

#: The file inside the storage directory that holds the records.
FILENAME = "events.jsonl"

#: How long an appending process waits for the lock before writing anyway.
#: Losing the lock is better than losing the record: this is monitoring.
LOCK_TIMEOUT = 10.0

#: Windows locks this byte, which is far past the end of any real log, so the
#: mandatory lock never overlaps a region a reader is reading.
_LOCK_OFFSET = 1 << 40

_IS_WINDOWS = os.name == "nt"

if _IS_WINDOWS:  # pragma: no cover - platform specific
    import msvcrt
else:  # pragma: no cover - platform specific
    try:
        import fcntl
    except ImportError:  # pragma: no cover - a POSIX without fcntl
        fcntl = None  # type: ignore[assignment]


def _lock_windows(handle: int, timeout: float) -> bool:  # pragma: no cover - platform specific
    deadline = time.monotonic() + timeout
    delay = 0.001
    while True:
        try:
            os.lseek(handle, _LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(delay)
            delay = min(delay * 2.0, 0.05)


def _unlock_windows(handle: int) -> None:  # pragma: no cover - platform specific
    try:
        os.lseek(handle, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def _lock_posix(handle: int, timeout: float) -> bool:  # pragma: no cover - platform specific
    if fcntl is None:
        return False
    deadline = time.monotonic() + timeout
    delay = 0.001
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(delay)
            delay = min(delay * 2.0, 0.05)


def _unlock_posix(handle: int) -> None:  # pragma: no cover - platform specific
    if fcntl is None:
        return
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def _exclusive(handle: int, path: Any) -> Iterator[bool]:
    """Hold an exclusive lock on ``handle`` for the body of the block.

    Yields whether the lock was actually taken. A lock that could not be taken
    is a warning, never an error: the record is still written.
    """
    locker = _lock_windows if _IS_WINDOWS else _lock_posix
    unlocker = _unlock_windows if _IS_WINDOWS else _unlock_posix
    locked = locker(handle, LOCK_TIMEOUT)
    if not locked:
        logger.warning(
            "model-watchdog: could not lock %s within %.0fs; appending anyway",
            path,
            LOCK_TIMEOUT,
        )
    try:
        yield locked
    finally:
        if locked:
            unlocker(handle)


class JsonlStorage:
    """Records on disk, one JSON object per line, in ``directory/events.jsonl``."""

    def __init__(self, directory: Any) -> None:
        self.directory = Path(directory)
        self.path = self.directory / FILENAME
        #: Set once the directory is known to exist, so the hot path skips a
        #: mkdir syscall per record. A directory that disappears afterwards is
        #: recreated on the next append rather than costing the record.
        self._directory_ready = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "JsonlStorage(%r)" % str(self.directory)

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one record. The directory is created on demand.

        Safe to call from several processes at once: the line is written whole,
        under an exclusive lock, to a descriptor opened for append.
        """
        line = json.dumps(dict(record), ensure_ascii=False, default=jsonable) + "\n"
        payload = line.encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        if not self._directory_ready:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._directory_ready = True
        try:
            handle = os.open(str(self.path), flags, 0o644)
        except OSError:
            # The directory went away under us (a tidy-up job, a rotated
            # volume). Recreate it and try once more before losing the record.
            self._directory_ready = False
            self.directory.mkdir(parents=True, exist_ok=True)
            self._directory_ready = True
            handle = os.open(str(self.path), flags, 0o644)
        try:
            with _exclusive(handle, self.path):
                # O_APPEND puts the write at end-of-file on every platform; the
                # explicit seek keeps Windows honest after the lock seek above.
                os.lseek(handle, 0, os.SEEK_END)
                written = 0
                while written < len(payload):
                    written += os.write(handle, payload[written:])
        finally:
            os.close(handle)

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """Every parseable record, oldest first. Bad lines are skipped."""
        if not self.path.exists():
            return
        skipped = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                if isinstance(record, dict):
                    yield record
                else:
                    skipped += 1
        if skipped:
            logger.warning(
                "model-watchdog: skipped %d unreadable line(s) in %s "
                "(partial write or hand-edited file)",
                skipped,
                self.path,
            )

    def read(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """The last ``limit`` records (or all of them), oldest first.

        Raises:
            TypeError: ``limit`` is neither ``None`` nor an integer. Comparing
                it to zero instead surfaces a bare "'<=' not supported between
                instances of 'str' and 'int'" from this layer, which names
                nothing the caller actually wrote.
            ValueError: ``limit`` is zero or negative. Reading no records and
                presenting that as a clean report is a lie about a log that has
                records in it.
        """
        if limit is None:
            return list(self.iter_records())
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(
                "limit must be a positive integer or None, got %s" % type(limit).__name__
            )
        if limit <= 0:
            raise ValueError("limit must be a positive integer or None, got %d" % limit)
        return list(deque(self.iter_records(), maxlen=int(limit)))

    def count(self) -> int:
        """How many records are readable right now."""
        return sum(1 for _ in self.iter_records())

    def exists(self) -> bool:
        """True once anything has been logged."""
        return self.path.exists()
