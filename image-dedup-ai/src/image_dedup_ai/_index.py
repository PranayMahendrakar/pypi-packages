"""The SQLite-backed index: hash each image once, re-check a folder instantly afterwards."""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from . import _hashing, _search
from ._hashing import IMAGE_EXTENSIONS, METHODS, Fingerprint, check_hash_size, check_method
from ._result import _HASH_LIMIT_NOTE, AddReport, DedupeResult, Match, Member

log = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]
EmbedFn = Callable[[Image.Image], Any]

ALL_METHODS: Tuple[str, ...] = METHODS + ("embed",)
SCHEMA_VERSION = "1"
_APP = "image-dedup-ai"
_BATCH = 256
_TOP = "\U0010ffff"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,
    bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    format TEXT NOT NULL,
    digest TEXT NOT NULL,
    flat INTEGER NOT NULL,
    tone REAL NOT NULL,
    phash BLOB NOT NULL,
    dhash BLOB NOT NULL,
    ahash BLOB NOT NULL,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS skipped (
    path TEXT PRIMARY KEY,
    bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    reason TEXT NOT NULL
);
"""

_INSERT_IMAGE = (
    "INSERT OR REPLACE INTO images (path, bytes, mtime_ns, width, height, format, digest, flat, tone, "
    "phash, dhash, ahash, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_INSERT_SKIPPED = "INSERT OR REPLACE INTO skipped (path, bytes, mtime_ns, reason) VALUES (?, ?, ?, ?)"


class _Table(NamedTuple):
    paths: List[str]
    bytes: np.ndarray
    width: np.ndarray
    height: np.ndarray
    digest: np.ndarray
    flat: np.ndarray
    data: np.ndarray  # (n, nbytes) uint8 hashes, or (n, dim) float32 unit vectors for "embed"


def _encodable(path: str) -> bool:
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:  # undecodable bytes in a file name (surrogate escapes)
        return False
    return True


def _shown(path: str) -> str:
    """``path`` itself, or a printable stand-in when it is not valid Unicode."""
    return path if _encodable(path) else path.encode("utf-8", "backslashreplace").decode("utf-8")


def _check_threshold(threshold: Any) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating, np.integer)):
        raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
    value = float(threshold)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    return value


def _check_workers(workers: Any) -> int:
    if workers is None:
        return max(1, min(8, os.cpu_count() or 1))
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or int(workers) < 1:
        raise ValueError(f"workers must be a positive int or None, got {workers!r}")
    return int(workers)


def _hash_path(path: str, hash_size: int) -> Tuple[Optional[Fingerprint], str]:
    """Fingerprint one file; never raises (an unreadable file is a reason, not a crash)."""
    try:
        return _hashing.fingerprint_file(path, hash_size), ""
    except ValueError as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"cannot read file ({exc.strerror or exc})"
    except MemoryError:
        return None, "out of memory while decoding"
    except Exception as exc:  # a broken codec must not stop a 100,000-file scan
        return None, f"unreadable image ({type(exc).__name__}: {exc})"


def _as_vector(value: Any) -> bytes:
    vec = np.asarray(value, dtype=np.float64).ravel()
    if vec.size == 0:
        raise ValueError("embed returned an empty vector")
    if not np.all(np.isfinite(vec)):
        raise ValueError("embed returned NaN or infinite values")
    return vec.astype("<f4").tobytes()


def _embed_id(embed: Any) -> str:
    name = getattr(embed, "__qualname__", None) or type(embed).__qualname__
    return f"{getattr(embed, '__module__', None) or type(embed).__module__}.{name}"


def _keep_key(i: int, t: "_Table") -> Tuple[int, int, int, str]:
    """Sort key for which copy to keep: largest resolution, then largest file,
    then the shortest path (``photo.jpg`` over ``photo - Copy.jpg``), then alphabetical."""
    return (-int(t.width[i]) * int(t.height[i]), -int(t.bytes[i]), len(t.paths[i]), t.paths[i])


def _pick_keep(members: Sequence[int], t: "_Table") -> int:
    return min(members, key=lambda i: _keep_key(i, t))


class Index:
    """A persistent index of image hashes.

    ``path`` is the SQLite file that keeps the hashes (``None`` keeps them in memory
    for this session only). Each file is keyed on its path, size and modification
    time, so adding the same folder again only reads new or edited files.

    ``hash_size`` sets the hash length (``hash_size ** 2`` bits, 64 by default).
    ``embed`` is an optional ``callable(PIL.Image) -> vector``; when given, every
    added image is also embedded (and the vector stored) so that
    ``find_duplicates(method="embed")`` can catch what a hash cannot, such as
    rotated, flipped or cropped copies. It receives an upright RGB copy.

    Use it as a context manager (``with Index("photos.idx") as idx:``) or call
    ``close()`` when done.
    """

    def __init__(self, path: Optional[PathLike] = None, *, hash_size: int = 8, embed: Optional[EmbedFn] = None) -> None:
        self.hash_size = check_hash_size(hash_size)
        if embed is not None and not callable(embed):
            raise TypeError("embed must be a callable that takes a PIL image and returns a vector")
        self.embed = embed
        self.bits = self.hash_size ** 2
        self.last_add = AddReport()
        self._version = 0
        self._cache: Dict[str, Tuple[int, _Table]] = {}
        self._conn: Optional[sqlite3.Connection] = None
        if path is None:
            self.path: Optional[str] = None
            target = ":memory:"
        else:
            target = os.path.abspath(os.fspath(path))
            if os.path.isdir(target):
                raise ValueError(
                    f"{target!r} is a folder. Index(path) is the file that keeps the hashes, "
                    "e.g. Index('photos.idx'); add the folder with idx.add(folder)."
                )
            parent = os.path.dirname(target)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            self.path = target
        conn = sqlite3.connect(target, timeout=30.0)
        try:
            conn.execute("PRAGMA synchronous = NORMAL")
            self._setup(conn, target)
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise ValueError(f"{target!r} is not an image-dedup-ai index ({exc})") from None
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    # ------------------------------------------------------------------ setup
    def _setup(self, conn: sqlite3.Connection, target: str) -> None:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if tables:
            meta: Dict[str, str] = {}
            if "meta" in tables:
                meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            if meta.get("app") != _APP:
                raise ValueError(f"{target!r} is a SQLite database but not an image-dedup-ai index; it was left untouched")
            if meta.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"{target!r} was written by an incompatible version (schema {meta.get('schema')!r})")
            stored = int(meta.get("hash_size", "0"))
            if stored != self.hash_size:
                raise ValueError(
                    f"{target!r} was built with hash_size={stored}; open it with Index(path, hash_size={stored})"
                )
        else:
            conn.executescript(_SCHEMA)
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [("app", _APP), ("schema", SCHEMA_VERSION), ("hash_size", str(self.hash_size)), ("image_counter", "0")],
            )
            meta = {}
        if self.embed is not None:
            new_id = _embed_id(self.embed)
            old_id = meta.get("embed_id")
            if old_id and old_id != new_id:
                log.info("embed function changed (%s -> %s); stored embeddings are dropped", old_id, new_id)
                conn.execute("UPDATE images SET embedding = NULL")
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_id', ?)", (new_id,))
        conn.commit()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("this Index is closed")
        return self._conn

    def _changed(self) -> None:
        self._version += 1
        self._cache.clear()

    # --------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Save and close the index. Safe to call twice."""
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        state = "closed" if self._conn is None else f"{self.count():,} images"
        return f"Index({self.path!r}, hash_size={self.hash_size}, {state})"

    def count(self) -> int:
        """Number of images in the index."""
        return int(self._db().execute("SELECT COUNT(*) FROM images").fetchone()[0])

    # --------------------------------------------------------------------- add
    def add(self, images: Any, *, recursive: bool = True, workers: Optional[int] = None) -> int:
        """Hash images into the index and return how many of them are now indexed.

        ``images`` is a folder, an image path, a ``PIL.Image``, a dict of
        ``{name: PIL.Image}``, or a list mixing paths, folders and images. Folders are
        scanned for image extensions (``recursive`` descends into subfolders). Files
        already indexed with the same size and modification time are not read
        again. Unreadable or non-image files are skipped, never raised: they are
        listed in ``idx.last_add.skipped`` and in ``DedupeResult.skipped``. Files
        without an image extension inside a folder are ignored and counted in
        ``idx.last_add.ignored``. Indexed files under a scanned folder that no longer
        exist are removed. Nothing passed in, on disk or in memory, is modified.
        """
        conn = self._db()
        n_workers = _check_workers(workers)
        report = AddReport()
        files, memory, folders = self._collect(images, recursive, report)

        known = {
            row[0]: (row[1], row[2], bool(row[3]))
            for row in conn.execute("SELECT path, bytes, mtime_ns, embedding IS NOT NULL FROM images")
        }
        known_skipped = {row[0]: (row[1], row[2], row[3]) for row in conn.execute("SELECT path, bytes, mtime_ns, reason FROM skipped")}

        to_hash: List[Tuple[str, int, int]] = []
        to_embed: List[str] = []
        for path in files:
            try:
                st = os.stat(path)
            except OSError as exc:
                self._forget(conn, path, report, f"cannot read file ({exc.strerror or exc})")
                continue
            if not stat.S_ISREG(st.st_mode):
                self._forget(conn, path, report, "not a regular file")
                continue
            size, mtime = int(st.st_size), int(st.st_mtime_ns)
            row = known.get(path)
            if row is not None and row[0] == size and row[1] == mtime:
                if self.embed is None or row[2]:
                    report.unchanged += 1
                else:
                    to_embed.append(path)
                continue
            srow = known_skipped.get(path)
            if srow is not None and srow[0] == size and srow[1] == mtime:
                report.skipped.append((path, srow[2]))
                continue
            to_hash.append((path, size, mtime))

        if to_hash:
            self._hash_files(conn, to_hash, n_workers, report)
        for path in to_embed:
            self._embed_existing(conn, path, report)
        for name, img in memory:
            self._add_memory(conn, name, img, report)
        for folder, walked_recursive, seen in folders:
            self._prune(conn, folder, walked_recursive, seen, report)
        conn.commit()
        self._changed()
        self.last_add = report
        log.info("image-dedup-ai: %s", report.summary())
        return report.indexed

    def _collect(
        self, images: Any, recursive: bool, report: AddReport
    ) -> Tuple[List[str], List[Tuple[str, Image.Image]], List[Tuple[str, bool, set]]]:
        files: List[str] = []
        memory: List[Tuple[str, Image.Image]] = []
        folders: List[Tuple[str, bool, set]] = []
        seen_files: set = set()

        def add_file(path: str) -> None:
            if path in seen_files:
                return
            seen_files.add(path)
            if not _encodable(path):
                report.skipped.append((_shown(path), "the file or folder name is not valid Unicode; rename it to index the file"))
                return
            files.append(path)

        def add_path(raw: Any, *, top_level: bool) -> None:
            path = os.path.abspath(os.fspath(raw))
            if os.path.isdir(path):
                found: set = set()
                for f in self._walk(path, recursive, report):
                    found.add(f)
                    add_file(f)
                if _encodable(path):
                    folders.append((path, recursive, found))
            elif os.path.exists(path):
                add_file(path)
            elif top_level:
                raise FileNotFoundError(f"{path!r} does not exist")
            else:
                self._forget(self._db(), path, report, "no such file or folder")

        if isinstance(images, (str, os.PathLike)):
            add_path(images, top_level=True)
        elif isinstance(images, Image.Image):
            memory.append((self._next_name(), images))
        elif isinstance(images, Mapping):
            for name, img in images.items():
                if not isinstance(img, Image.Image):
                    raise TypeError(f"dict values must be PIL images; {name!r} is {type(img).__name__}")
                memory.append((str(name), img))
        elif isinstance(images, Iterable):
            for item in images:
                if isinstance(item, (str, os.PathLike)):
                    add_path(item, top_level=False)
                elif isinstance(item, Image.Image):
                    memory.append((self._next_name(), item))
                else:
                    raise TypeError(f"cannot add {type(item).__name__}; pass paths, folders or PIL images")
        else:
            raise TypeError(f"cannot add {type(images).__name__}; pass a folder, paths or PIL images")
        return files, memory, folders

    def _walk(self, top: str, recursive: bool, report: AddReport) -> List[str]:
        own = set()
        if self.path:
            own = {self.path, self.path + "-journal", self.path + "-wal", self.path + "-shm"}
        out: List[str] = []

        def on_error(exc: OSError) -> None:
            report.skipped.append((_shown(str(exc.filename or top)), f"cannot list folder ({exc.strerror or exc})"))

        if recursive:
            walker = os.walk(top, onerror=on_error)
        else:
            try:
                names = os.listdir(top)
            except OSError as exc:
                on_error(exc)
                names = []
            walker = iter([(top, [], [n for n in names if os.path.isfile(os.path.join(top, n))])])
        for dirpath, dirnames, filenames in walker:
            dirnames.sort()
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if full in own:
                    continue
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                    out.append(full)
                else:
                    report.ignored.append(_shown(full))
        return out

    def _next_name(self) -> str:
        conn = self._db()
        row = conn.execute("SELECT value FROM meta WHERE key = 'image_counter'").fetchone()
        n = int(row[0]) + 1 if row else 1
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('image_counter', ?)", (str(n),))
        return f"<image {n}>"

    def _forget(self, conn: sqlite3.Connection, path: str, report: AddReport, reason: str) -> None:
        """A path that cannot be indexed right now: report it and drop any stale entry."""
        report.skipped.append((_shown(path), reason))
        if not _encodable(path):
            return
        cur = conn.execute("DELETE FROM images WHERE path = ?", (path,))
        if cur.rowcount:
            report.removed += cur.rowcount
        conn.execute("DELETE FROM skipped WHERE path = ?", (path,))
        log.debug("skipped %s: %s", path, reason)

    def _embed_vector(self, image: Image.Image) -> bytes:
        assert self.embed is not None
        return _as_vector(self.embed(image))

    def _embed_file(self, path: str) -> Tuple[Optional[bytes], str]:
        try:
            upright = _hashing.open_upright_rgb(path)
        except Exception as exc:
            return None, f"cannot decode for embed ({type(exc).__name__}: {exc})"
        try:
            return self._embed_vector(upright), ""
        except Exception as exc:
            return None, f"embed failed ({type(exc).__name__}: {exc})"

    def _record_skip(self, conn: sqlite3.Connection, path: str, size: int, mtime: int, reason: str, report: AddReport) -> None:
        conn.execute("DELETE FROM images WHERE path = ?", (path,))
        conn.execute(_INSERT_SKIPPED, (path, size, mtime, reason))
        report.skipped.append((path, reason))
        log.debug("skipped %s: %s", path, reason)

    def _hash_files(self, conn: sqlite3.Connection, todo: List[Tuple[str, int, int]], workers: int, report: AddReport) -> None:
        pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 and len(todo) >= 8 else None
        try:
            for start in range(0, len(todo), _BATCH):
                batch = todo[start:start + _BATCH]
                paths = [p for p, _, _ in batch]
                if pool is not None:
                    results = list(pool.map(_hash_path, paths, [self.hash_size] * len(paths)))
                else:
                    results = [_hash_path(p, self.hash_size) for p in paths]
                for (path, size, mtime), (fp, reason) in zip(batch, results):
                    if fp is None:
                        self._record_skip(conn, path, size, mtime, reason, report)
                        continue
                    vector: Optional[bytes] = None
                    if self.embed is not None:
                        vector, reason = self._embed_file(path)
                        if vector is None:
                            self._record_skip(conn, path, size, mtime, reason, report)
                            continue
                        report.embedded += 1
                    conn.execute(_INSERT_IMAGE, self._row(path, size, mtime, fp, vector))
                    conn.execute("DELETE FROM skipped WHERE path = ?", (path,))
                    report.hashed += 1
                conn.commit()
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    def _embed_existing(self, conn: sqlite3.Connection, path: str, report: AddReport) -> None:
        vector, reason = self._embed_file(path)
        if vector is None:
            row = conn.execute("SELECT bytes, mtime_ns FROM images WHERE path = ?", (path,)).fetchone()
            size, mtime = (row[0], row[1]) if row else (0, 0)
            self._record_skip(conn, path, size, mtime, reason, report)
            return
        conn.execute("UPDATE images SET embedding = ? WHERE path = ?", (vector, path))
        report.embedded += 1
        report.unchanged += 1

    def _add_memory(self, conn: sqlite3.Connection, name: str, img: Image.Image, report: AddReport) -> None:
        try:
            fp = _hashing.fingerprint_image(img, self.hash_size)
        except Exception as exc:
            conn.execute("DELETE FROM images WHERE path = ?", (name,))
            report.skipped.append((name, f"unreadable image ({type(exc).__name__}: {exc})"))
            return
        vector: Optional[bytes] = None
        if self.embed is not None:
            try:
                vector = self._embed_vector(_hashing.upright_rgb(img))
            except Exception as exc:
                conn.execute("DELETE FROM images WHERE path = ?", (name,))
                report.skipped.append((name, f"embed failed ({type(exc).__name__}: {exc})"))
                return
            report.embedded += 1
        conn.execute(_INSERT_IMAGE, self._row(name, 0, -1, fp, vector))
        report.hashed += 1

    @staticmethod
    def _row(path: str, size: int, mtime: int, fp: Fingerprint, vector: Optional[bytes]) -> tuple:
        return (
            path, size, mtime, fp.width, fp.height, fp.format, fp.digest, int(fp.flat), fp.tone,
            fp.hashes["phash"], fp.hashes["dhash"], fp.hashes["ahash"], vector,
        )

    def _under(self, conn: sqlite3.Connection, table: str, folder: str) -> List[str]:
        prefix = folder.rstrip("\\/") + os.sep
        rows = conn.execute(f"SELECT path FROM {table} WHERE path >= ? AND path < ?", (prefix, prefix + _TOP))
        return [r[0] for r in rows if r[0].startswith(prefix)]

    def _prune(self, conn: sqlite3.Connection, folder: str, walked_recursive: bool, seen: set, report: AddReport) -> None:
        """Drop entries under a scanned folder whose files are gone from disk."""
        for table in ("images", "skipped"):
            gone = [p for p in self._under(conn, table, folder) if p not in seen and not os.path.exists(p)]
            if gone:
                conn.executemany(f"DELETE FROM {table} WHERE path = ?", [(p,) for p in gone])
                if table == "images":
                    report.removed += len(gone)

    # ------------------------------------------------------------------ remove
    def remove(self, path: PathLike) -> int:
        """Remove an image (or every image under a folder) from the index; returns how many."""
        conn = self._db()
        raw = os.fspath(path)
        if not _encodable(raw):
            return 0  # such a path can never have been indexed
        removed = conn.execute("DELETE FROM images WHERE path = ?", (raw,)).rowcount
        conn.execute("DELETE FROM skipped WHERE path = ?", (raw,))
        if not removed:
            full = os.path.abspath(raw)
            removed = conn.execute("DELETE FROM images WHERE path = ?", (full,)).rowcount
            conn.execute("DELETE FROM skipped WHERE path = ?", (full,))
            if not removed:
                for table in ("images", "skipped"):
                    under = self._under(conn, table, full)
                    conn.executemany(f"DELETE FROM {table} WHERE path = ?", [(p,) for p in under])
                    if table == "images":
                        removed = len(under)
        conn.commit()
        self._changed()
        return int(removed)

    # ------------------------------------------------------------------- query
    def _load(self, method: str) -> _Table:
        cached = self._cache.get(method)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        conn = self._db()
        column = "embedding" if method == "embed" else method  # validated: one of ALL_METHODS
        where = " WHERE embedding IS NOT NULL" if method == "embed" else ""
        rows = conn.execute(
            f"SELECT path, bytes, width, height, digest, flat, {column} FROM images{where} ORDER BY path"
        ).fetchall()
        n = len(rows)
        paths = [r[0] for r in rows]
        if method == "embed":
            dims = sorted({len(r[6]) // 4 for r in rows})
            if len(dims) > 1:
                raise ValueError(
                    f"stored embeddings have different lengths {dims}; the embed function changed, use a new index file"
                )
            dim = dims[0] if dims else 0
            data = np.frombuffer(b"".join(r[6] for r in rows), dtype="<f4").reshape(n, dim).astype(np.float32)
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            data = np.divide(data, norms, out=np.zeros_like(data), where=norms > 0)
        else:
            nbytes = (self.bits + 7) // 8
            data = np.frombuffer(b"".join(r[6] for r in rows), dtype=np.uint8).reshape(n, nbytes).copy()
        table = _Table(
            paths=paths,
            bytes=np.array([r[1] for r in rows], dtype=np.int64),
            width=np.array([r[2] for r in rows], dtype=np.int64),
            height=np.array([r[3] for r in rows], dtype=np.int64),
            digest=np.array([r[4] for r in rows], dtype=object),
            flat=np.array([bool(r[5]) for r in rows], dtype=bool),
            data=data,
        )
        self._cache[method] = (self._version, table)
        return table

    def _skipped(self) -> List[Tuple[str, str]]:
        """Files known to be unreadable, plus anything the last ``add`` could not even try."""
        found = {r[0]: r[1] for r in self._db().execute("SELECT path, reason FROM skipped")}
        for path, reason in self.last_add.skipped:
            found.setdefault(path, reason)
        return sorted(found.items())

    def _hash_edges(self, t: _Table, max_dist: int) -> Tuple[List[int], List[int], List[float]]:
        """Similarity links that form the groups, kept linear in the number of copies.

        Byte-identical files are linked to the best copy among them; distinct files
        with an identical hash are linked to the best of those; then each pair of
        distinct hashes within ``max_dist`` bits is linked once, best copy to best copy.
        Near-blank (flat) images only take part in the first, identical-content step.
        """
        n = len(t.paths)
        ei: List[int] = []
        ej: List[int] = []
        es: List[float] = []
        if n < 2:
            return ei, ej, es
        rank = np.empty(n, dtype=np.int64)
        rank[sorted(range(n), key=lambda i: _keep_key(i, t))] = np.arange(n)

        def star(rows: np.ndarray, labels: np.ndarray) -> np.ndarray:
            """Link each row to the best-ranked row with the same label; return those best rows."""
            order = np.argsort(rank[rows], kind="stable")
            rows, labels = rows[order], labels[order]
            _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
            inverse = np.asarray(inverse).ravel()
            reps = rows[first]
            others = np.flatnonzero(rows != reps[inverse])
            ei.extend(reps[inverse[others]].tolist())
            ej.extend(rows[others].tolist())
            es.extend([1.0] * int(others.size))
            return reps

        distinct_content = star(np.arange(n, dtype=np.int64), t.digest.astype(str))
        detailed = distinct_content[~t.flat[distinct_content]]
        if detailed.size >= 2:
            packed = np.ascontiguousarray(t.data[detailed])
            keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
            reps = star(detailed, keys)
            ci, cj, d = _search.similar_pairs(np.ascontiguousarray(t.data[reps]), self.bits, max_dist)
            ei.extend(reps[ci].tolist())
            ej.extend(reps[cj].tolist())
            es.extend((1.0 - d / float(self.bits)).tolist())
        return ei, ej, es

    def _similarity_to(self, t: _Table, keep: int, members: Sequence[int], method: str) -> Dict[int, float]:
        if method == "embed":
            return {m: float(np.clip(t.data[m] @ t.data[keep], -1.0, 1.0)) for m in members}
        words = _search.to_words(t.data[list(members)])
        query = _search.to_words(t.data[keep:keep + 1])[0]
        dist = _search.distances_to(words, query)
        out = {}
        for m, d in zip(members, dist.tolist()):
            out[m] = 1.0 if (t.flat[m] and t.flat[keep]) else 1.0 - d / float(self.bits)
        return out

    def _build_result(
        self, t: _Table, ei: List[int], ej: List[int], es: List[float], method: str, threshold: float, notes: List[str]
    ) -> DedupeResult:
        n = len(t.paths)
        labels = _search.components(n, np.asarray(ei, dtype=np.int64), np.asarray(ej, dtype=np.int64))
        counts = np.bincount(labels, minlength=n) if n else np.zeros(0, dtype=np.int64)
        members_of: Dict[int, List[int]] = {}
        for i in np.flatnonzero(counts[labels] >= 2).tolist() if n else []:
            members_of.setdefault(int(labels[i]), []).append(i)

        built = []
        for members in members_of.values():
            keep = _pick_keep(members, t)
            sims = self._similarity_to(t, keep, members, method)
            others = sorted(
                (m for m in members if m != keep),
                key=lambda m: (t.digest[m] != t.digest[keep], -sims[m], t.paths[m]),
            )
            detail = [
                Member(
                    path=t.paths[m],
                    width=int(t.width[m]),
                    height=int(t.height[m]),
                    bytes=int(t.bytes[m]),
                    similarity=1.0 if m == keep else round(float(sims[m]), 6),
                    identical=bool(t.digest[m] == t.digest[keep]),
                    kept=m == keep,
                )
                for m in [keep] + others
            ]
            wasted = int(sum(int(t.bytes[m]) for m in others))
            built.append((wasted, detail))
        built.sort(key=lambda item: (-item[0], -len(item[1]), item[1][0].path))

        pairs = []
        for a, b, s in zip(ei, ej, es):
            pa, pb = sorted((t.paths[a], t.paths[b]))
            pairs.append((pa, pb, round(float(s), 6)))
        pairs.sort(key=lambda p: (-p[2], p[0], p[1]))

        return DedupeResult(
            groups=[[m.path for m in detail] for _, detail in built],
            pairs=pairs,
            keep=[detail[0].path for _, detail in built],
            wasted_bytes=int(sum(w for w, _ in built)),
            method=method,
            threshold=threshold,
            n_images=self.count(),
            bits=0 if method == "embed" else self.bits,
            details=[detail for _, detail in built],
            skipped=self._skipped(),
            notes=notes,
        )

    def find_duplicates(self, *, threshold: float = 0.9, method: str = "phash") -> DedupeResult:
        """Group the indexed images whose similarity is at least ``threshold``.

        ``method`` is ``"phash"`` (default, robust to resizing and re-compression),
        ``"dhash"`` (gradient direction), ``"ahash"`` (mean; fastest and weakest) or
        ``"embed"`` (cosine similarity of the vectors from ``Index(embed=...)``).
        Hash similarity is ``1 - hamming / bits``. Groups are the connected
        components of all pairs at or above the threshold.
        """
        threshold = _check_threshold(threshold)
        method = check_method(method, ALL_METHODS)
        if method == "embed":
            return self._find_embed(threshold)
        t = self._load(method)
        max_dist = int(math.floor((1.0 - threshold) * self.bits + 1e-9))
        ei, ej, es = self._hash_edges(t, max_dist)
        notes: List[str] = []
        n_flat = int(t.flat.sum())
        if n_flat:
            notes.append(
                f"{n_flat:,} near-blank {'image has' if n_flat == 1 else 'images have'} almost no detail, "
                "so a hash says nothing about them; they are grouped only with identical content."
            )
        if threshold < 0.8:
            notes.append(f"threshold {threshold:g} is loose: below about 0.8, unrelated images start to match.")
        if t.paths:
            notes.append(_HASH_LIMIT_NOTE)
        return self._build_result(t, ei, ej, es, method, threshold, notes)

    def _find_embed(self, threshold: float) -> DedupeResult:
        total = self.count()
        t = self._load("embed")
        if total and not t.paths:
            raise ValueError(
                "no embeddings in this index: create it with Index(..., embed=fn) and add the images, "
                "or use method='phash'"
            )
        i, j, s = _search.cosine_pairs(t.data, threshold)
        notes: List[str] = []
        missing = total - len(t.paths)
        if missing:
            notes.append(f"{missing:,} indexed images have no embedding and were not compared (add them with embed= set).")
        if t.paths:
            notes.append("Embedding similarity depends on the model you passed as embed=; tune threshold for it.")
        return self._build_result(t, i.tolist(), j.tolist(), s.tolist(), "embed", threshold, notes)

    # -------------------------------------------------------------------- near
    def _query(self, image: Any, method: str) -> Tuple[Any, bool, str, Optional[str]]:
        """(hash bytes or vector, flat, digest, indexed path or None) for a query image."""
        if isinstance(image, Image.Image):
            if method == "embed":
                if self.embed is None:
                    raise ValueError("near(..., method='embed') on a PIL image needs Index(embed=fn)")
                return np.frombuffer(self._embed_vector(_hashing.upright_rgb(image)), dtype="<f4"), False, "", None
            fp = _hashing.fingerprint_image(image, self.hash_size)
            return fp.hashes[method], fp.flat, fp.digest, None
        if not isinstance(image, (str, os.PathLike)):
            raise TypeError(f"near() takes an image path or a PIL image, got {type(image).__name__}")
        raw = os.fspath(image)
        conn = self._db()
        row = None
        if _encodable(raw):
            row = conn.execute(
                "SELECT path, bytes, mtime_ns, flat, digest, phash, dhash, ahash, embedding FROM images "
                "WHERE path = ? OR path = ?",
                (raw, os.path.abspath(raw)),
            ).fetchone()
        if row is not None:
            fresh = row[2] < 0
            if not fresh:
                try:
                    st = os.stat(row[0])
                    fresh = (st.st_size, st.st_mtime_ns) == (row[1], row[2])
                except OSError:
                    fresh = False
            if fresh:
                if method == "embed":
                    if row[8] is not None:
                        return np.frombuffer(row[8], dtype="<f4"), False, row[4], row[0]
                else:
                    blob = {"phash": row[5], "dhash": row[6], "ahash": row[7]}[method]
                    return blob, bool(row[3]), row[4], row[0]
        path = os.path.abspath(raw)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path!r} does not exist")
        if method == "embed":
            if self.embed is None:
                raise ValueError("near(..., method='embed') on a file that is not embedded yet needs Index(embed=fn)")
            vector, reason = self._embed_file(path)
            if vector is None:
                raise ValueError(f"cannot embed {path!r}: {reason}")
            return np.frombuffer(vector, dtype="<f4"), False, "", path
        fp, reason = _hash_path(path, self.hash_size)
        if fp is None:
            raise ValueError(f"cannot read {path!r}: {reason}")
        return fp.hashes[method], fp.flat, fp.digest, path

    def near(self, image: Any, *, k: int = 5, method: str = "phash") -> List[Match]:
        """The ``k`` indexed images most similar to ``image`` (a path or ``PIL.Image``).

        Returns ``Match(path, similarity)`` tuples, most similar first. The query
        itself is left out when it is one of the indexed files. The query is hashed
        but not added to the index.
        """
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        method = check_method(method, ALL_METHODS)
        t = self._load(method)
        n = len(t.paths)
        if method == "embed" and n == 0 and self.count():
            raise ValueError("no embeddings in this index: create it with Index(..., embed=fn) and add the images")
        query, q_flat, q_digest, q_path = self._query(image, method)
        if n == 0:
            return []
        if method == "embed":
            q = np.asarray(query, dtype=np.float32)
            if q.shape[0] != t.data.shape[1]:
                raise ValueError(f"query embedding has length {q.shape[0]}, the index has {t.data.shape[1]}")
            norm = float(np.linalg.norm(q))
            sims = (t.data @ (q / norm)).astype(np.float64) if norm > 0 else np.zeros(n)
            sims = np.clip(sims, -1.0, 1.0)
        else:
            words = _search.to_words(t.data)
            q_words = _search.to_words(np.frombuffer(query, dtype=np.uint8)[None, :])[0]
            sims = 1.0 - _search.distances_to(words, q_words) / float(self.bits)
            if q_flat:
                sims = np.where(t.flat & (t.digest == q_digest), 1.0, 0.0)
            else:
                sims = np.where(t.flat, 0.0, sims)
        order = np.lexsort((np.arange(n), -sims))
        out: List[Match] = []
        for i in order.tolist():
            if q_path is not None and t.paths[i] == q_path:
                continue
            out.append(Match(t.paths[i], round(float(sims[i]), 6)))
            if len(out) == int(k):
                break
        return out


def find_duplicates(
    folder: Any,
    *,
    threshold: float = 0.9,
    method: str = "phash",
    hash_size: int = 8,
    recursive: bool = True,
    index: Optional[PathLike] = None,
    embed: Optional[EmbedFn] = None,
    workers: Optional[int] = None,
) -> DedupeResult:
    """Find duplicate images in one call.

    ``folder`` is anything ``Index.add`` accepts: a folder, image paths, PIL images
    or a ``{name: image}`` dict. ``index`` is an optional file that keeps the
    hashes, so the next call only reads new or edited files. Other options are as
    for ``Index`` and ``Index.find_duplicates``.
    """
    with Index(index, hash_size=hash_size, embed=embed) as idx:
        idx.add(folder, recursive=recursive, workers=workers)
        result = idx.find_duplicates(threshold=threshold, method=method)
        note = ignored_note(idx.last_add)
        if note:
            result.notes.insert(0, note)
    return result


def ignored_note(report: AddReport) -> str:
    """A note for files a folder scan left out because they have no image extension."""
    n = len(report.ignored)
    if not n:
        return ""
    return f"{n:,} {'file' if n == 1 else 'files'} without an image extension {'was' if n == 1 else 'were'} ignored."
