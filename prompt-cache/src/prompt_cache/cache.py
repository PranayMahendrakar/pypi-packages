"""The cache itself: :class:`Cache` and the :func:`cached` decorator."""

from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ._keys import make_key, normalize_prompt, preview
from ._serialize import decode, encode
from ._store import MEMORY, Store, logger
from .stats import Stats

#: Folder used when no ``path`` is given.
DEFAULT_DIR = "~/.prompt_cache"

#: File created inside that folder (or inside any folder passed as ``path``).
DB_FILENAME = "cache.db"

#: Namespace used when none is given.
DEFAULT_NAMESPACE = "default"

#: How long a write waits for another process to finish, in seconds.
BUSY_TIMEOUT = 15.0

#: A ``path`` ending in one of these is taken to be the database file itself.
DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")

_MB = 1024.0 * 1024.0


def _now() -> float:
    """Current wall-clock time. Patched in tests instead of sleeping."""
    return time.time()


def resolve_path(path: Optional[Union[str, Path]] = None) -> Union[str, Path]:
    """Work out which file a ``path`` argument means.

    ``None`` means ``~/.prompt_cache/cache.db``. A path ending in ``.db``,
    ``.sqlite`` or ``.sqlite3`` is used as the database file; anything else is
    treated as a folder and gets ``cache.db`` inside it. ``":memory:"`` gives
    a private database that disappears with the process.
    """
    if path is None:
        path = DEFAULT_DIR
    if not isinstance(path, (str, Path)):
        raise TypeError(
            "path must be a str, a pathlib.Path or None, not {0!r}".format(type(path).__name__)
        )
    if str(path) == MEMORY:
        return MEMORY
    resolved = Path(path).expanduser()
    if resolved.suffix.lower() in DB_SUFFIXES:
        return resolved
    return resolved / DB_FILENAME


def _check_number(value: Any, name: str, *, allow_zero: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{0} must be a number of seconds or None, not {1!r}".format(name, type(value).__name__))
    number = float(value)
    if number != number:  # NaN
        raise ValueError("{0} must be a real number, not NaN".format(name))
    if number < 0 or (number == 0 and not allow_zero):
        limit = "0 or more" if allow_zero else "greater than 0"
        raise ValueError("{0} must be {1}, got {2!r}".format(name, limit, value))
    return number


def _first_parameter_name(func: Callable[..., Any]) -> Optional[str]:
    """The name of the first positional parameter, so it can be passed by keyword."""
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return None
    for parameter in parameters:
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            return parameter.name
    return None


class Cache:
    """A disk-backed cache of answers, keyed by prompt plus parameters.

    Args:
        path: Folder for the cache, or the database file itself. Defaults to
            ``~/.prompt_cache``. The folder is created the first time
            something is actually read or written.
        ttl: Seconds an entry stays valid. ``None`` means forever; ``0`` turns
            caching off entirely, so nothing is written and every lookup
            misses.
        max_size_mb: Soft ceiling on the megabytes of cached values in this
            namespace. When a write pushes it over, the least recently used
            entries are dropped until it fits again.
        namespace: Name for this logical cache. A cache only ever sees its own
            namespace, so several of them can share one file without
            colliding. Defaults to ``"default"``.
        normalize: Strip trailing whitespace and collapse blank lines before
            hashing, so prompts that differ only in layout share one entry.
            See :func:`prompt_cache.normalize_prompt` for the exact rules.

    Storage is SQLite in WAL mode with a busy timeout, so several processes
    may share one cache file safely.

    Example:
        >>> cache = Cache(":memory:")
        >>> cache.set("2 + 2?", "4", model="demo")
        >>> cache.get("2 + 2?", model="demo")
        '4'
        >>> cache.get("2 + 2?", model="other") is None
        True
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        ttl: Optional[float] = None,
        max_size_mb: Optional[float] = None,
        namespace: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self._ttl = _check_number(ttl, "ttl", allow_zero=True)
        self._max_size_mb = _check_number(max_size_mb, "max_size_mb", allow_zero=False)
        if namespace is None:
            namespace = DEFAULT_NAMESPACE
        if not isinstance(namespace, str):
            raise TypeError("namespace must be a str or None, not {0!r}".format(type(namespace).__name__))
        if not namespace:
            raise ValueError("namespace must not be empty")
        self._namespace = namespace
        self._normalize = bool(normalize)
        self._store = Store(resolve_path(path), timeout=BUSY_TIMEOUT)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._saved_calls = 0

    # ------------------------------------------------------------ properties

    @property
    def path(self) -> Union[str, Path]:
        """The database file this cache reads and writes."""
        return self._store.path

    @property
    def ttl(self) -> Optional[float]:
        """Seconds an entry stays valid, or ``None`` for forever."""
        return self._ttl

    @property
    def max_size_mb(self) -> Optional[float]:
        """Megabyte ceiling for this namespace, or ``None`` for no limit."""
        return self._max_size_mb

    @property
    def namespace(self) -> str:
        """The only namespace this cache can see."""
        return self._namespace

    @property
    def normalize(self) -> bool:
        """Whether prompts are normalised before hashing."""
        return self._normalize

    @property
    def enabled(self) -> bool:
        """False when ``ttl=0`` has switched caching off."""
        return self._ttl != 0

    # --------------------------------------------------------------- reading

    def key(self, prompt: str, **params: Any) -> str:
        """The database key for a prompt and its parameters (a SHA-256 digest)."""
        return self._key(prompt, params)

    def _key(
        self,
        prompt: str,
        params: Mapping[str, Any],
        args: Sequence[Any] = (),
        function: Optional[str] = None,
    ) -> str:
        return make_key(
            prompt,
            namespace=self._namespace,
            params=params,
            args=args,
            function=function,
            normalize=self._normalize,
        )

    @staticmethod
    def _check_prompt(prompt: Any) -> str:
        if not isinstance(prompt, str):
            raise TypeError(
                "prompt must be a str, not {0!r}; convert it first (for example "
                "with json.dumps for a message list)".format(type(prompt).__name__)
            )
        return prompt

    def _lookup(
        self,
        prompt: str,
        params: Mapping[str, Any],
        args: Sequence[Any] = (),
        function: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """``(found, value)`` - the only way to tell a stored ``None`` from a miss."""
        self._check_prompt(prompt)
        if not self.enabled:
            self._misses += 1
            return False, None
        key = self._key(prompt, params, args, function)
        row = self._store.lookup(key, _now())
        if row is None:
            self._misses += 1
            return False, None
        fmt, blob = row
        try:
            value = decode(fmt, blob)
        except Exception as exc:  # unreadable blob: treat as a miss, drop the row
            logger.warning("prompt-cache: dropping an unreadable entry (%s)", exc)
            self._store.delete(key)
            self._misses += 1
            return False, None
        self._hits += 1
        return True, value

    def lookup(self, prompt: str, **params: Any) -> Tuple[bool, Any]:
        """``(found, value)`` for this prompt - the honest version of :meth:`get`.

        Use it when ``None`` is a value you might legitimately have stored:
        ``found`` says whether the entry exists, ``value`` is what it holds.
        """
        return self._lookup(prompt, params)

    def get(self, prompt: str, **params: Any) -> Any:
        """Return the cached value for this prompt, or ``None``.

        Every keyword argument takes part in the key, so a different model or
        temperature is a different entry. A missing key costs one indexed
        SELECT and writes nothing.

        Note:
            A stored value of ``None`` is indistinguishable from a miss here.
            :meth:`wrap` does not have that problem: it checks whether the row
            exists, so a function that legitimately returns ``None`` is still
            cached.
        """
        found, value = self._lookup(prompt, params)
        return value if found else None

    # --------------------------------------------------------------- writing

    def set(self, prompt: str, value: Any, **params: Any) -> None:
        """Store ``value`` for this prompt and these parameters.

        Raises:
            TypeError: if ``prompt`` is not a string, or if ``value`` can be
                serialised by neither JSON nor pickle. The message names the
                type that could not be stored.
        """
        self._check_prompt(prompt)
        self._store_value(prompt, value, params)

    def _store_value(
        self,
        prompt: str,
        value: Any,
        params: Mapping[str, Any],
        args: Sequence[Any] = (),
        function: Optional[str] = None,
    ) -> None:
        fmt, blob = encode(value)  # raises TypeError naming the type
        if not self.enabled:
            return
        max_bytes = None
        if self._max_size_mb is not None:
            max_bytes = int(self._max_size_mb * _MB)
            if len(blob) > max_bytes:
                logger.warning(
                    "prompt-cache: value of %d bytes is larger than max_size_mb=%s; not cached",
                    len(blob),
                    self._max_size_mb,
                )
                return
        now = _now()
        expires = None if self._ttl is None else now + self._ttl
        self._store.store(
            self._key(prompt, params, args, function),
            self._namespace,
            fmt,
            blob,
            now,
            expires,
            preview(normalize_prompt(prompt) if self._normalize else prompt),
        )
        if max_bytes is not None:
            self._evictions += self._store.evict(self._namespace, max_bytes, now)

    # ------------------------------------------------------------ decorating

    def wrap(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Return ``func`` with a cache in front of it.

        The first argument is the prompt; every other argument takes part in
        the key, as does the function's module and qualified name, so two
        different functions never read each other's answers. An exception
        inside ``func`` is never cached. The wrapper carries the cache as
        ``wrapped.cache``.

        Example:
            >>> cache = Cache(":memory:")
            >>> @cache.wrap
            ... def ask(prompt, model="demo"):
            ...     return prompt.upper()
            >>> ask("hello"), ask("hello")
            ('HELLO', 'HELLO')
            >>> cache.stats().saved_calls
            1
        """
        if not callable(func):
            raise TypeError("wrap() needs a callable, not {0!r}".format(type(func).__name__))
        function = "{0}.{1}".format(getattr(func, "__module__", "?"), getattr(func, "__qualname__", repr(func)))
        first = _first_parameter_name(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if args:
                prompt, rest = args[0], args[1:]
            elif first is not None and first in kwargs:
                kwargs = dict(kwargs)
                prompt = kwargs.pop(first)
                rest = ()
            else:
                raise TypeError(
                    "{0}() is cached on its first argument, the prompt, "
                    "but none was given".format(getattr(func, "__name__", "function"))
                )
            found, value = self._lookup(prompt, kwargs, rest, function)
            if found:
                self._saved_calls += 1
                return value
            result = func(prompt, *rest, **kwargs)
            self._store_value(prompt, result, kwargs, rest, function)
            return result

        wrapper.cache = self  # type: ignore[attr-defined]
        return wrapper

    def memoize(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Alias of :meth:`wrap`, for when ``@cache.memoize`` reads better."""
        return self.wrap(func)

    # ---------------------------------------------------------- housekeeping

    def stats(self) -> Stats:
        """A :class:`~prompt_cache.Stats` snapshot of this cache."""
        entries, nbytes = self._store.totals(self._namespace, _now())
        lookups = self._hits + self._misses
        return Stats(
            hits=self._hits,
            misses=self._misses,
            hit_rate=(self._hits / lookups) if lookups else 0.0,
            entries=entries,
            size_mb=round(nbytes / _MB, 6),
            evictions=self._evictions,
            saved_calls=self._saved_calls,
        )

    def entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """The most recently used entries in this namespace, newest first.

        Each dict holds the key, a short prompt preview, the size in bytes,
        timestamps and the hit count. Prompts themselves are not stored in
        full: only the hashed key and a preview of at most 200 characters.
        """
        return self._store.entries(self._namespace, _now(), limit=max(int(limit), 0))

    def clear(self, namespace: Optional[str] = None) -> int:
        """Delete every entry in a namespace and return how many went.

        With no argument this clears the cache's own namespace. Pass
        ``namespace="*"`` to empty the whole file, including namespaces
        written by other caches.
        """
        if namespace is None:
            target: Optional[str] = self._namespace
        elif namespace == "*":
            target = None
        elif isinstance(namespace, str):
            target = namespace
        else:
            raise TypeError("namespace must be a str or None, not {0!r}".format(type(namespace).__name__))
        return self._store.clear(target)

    def prune(self) -> int:
        """Delete expired entries in this namespace and return how many went."""
        return self._store.prune(self._namespace, _now())

    def close(self) -> None:
        """Close the database connection.

        Safe to call twice. The cache reopens by itself if it is used again,
        so closing early can never break a caller.
        """
        self._store.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            "Cache(path={path!r}, ttl={ttl!r}, max_size_mb={size!r}, "
            "namespace={ns!r}, normalize={norm!r})".format(
                path=str(self.path),
                ttl=self._ttl,
                size=self._max_size_mb,
                ns=self._namespace,
                norm=self._normalize,
            )
        )


def cached(path: Optional[Union[str, Path]] = None, **kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that caches a function on its first argument.

    ``cached(...)`` builds one :class:`Cache` and hands it to every function
    it decorates; the wrapped function exposes it as ``func.cache``. Keyword
    arguments are the ones :class:`Cache` takes.

    Example:
        >>> @cached(":memory:", ttl=3600)
        ... def ask(prompt, model="demo"):
        ...     return "answer to " + prompt
        >>> ask("why?") == ask("why?")
        True
        >>> ask.cache.stats().hits
        1
    """
    cache = Cache(path, **kw)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return cache.wrap(func)

    decorator.cache = cache  # type: ignore[attr-defined]
    return decorator


__all__ = [
    "BUSY_TIMEOUT",
    "DB_FILENAME",
    "DB_SUFFIXES",
    "DEFAULT_DIR",
    "DEFAULT_NAMESPACE",
    "Cache",
    "cached",
    "resolve_path",
]
