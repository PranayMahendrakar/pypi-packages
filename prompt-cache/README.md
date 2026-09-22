# prompt-cache

Cache LLM answers on disk, so the second time your code sends the same prompt it costs nothing and comes back instantly.

## Install

```bash
pip install prompt-cache
```

No dependencies: `sqlite3`, `hashlib`, `json` and `pickle` are all in the standard library. Nothing is ever downloaded at runtime.

## Quickstart

```python
import prompt_cache

@prompt_cache.cached(".demo-cache")        # leave the path out to use ~/.prompt_cache
def ask(prompt, model="gpt-4o"):
    return "pretend this call cost you money: " + prompt

print(ask("Why is the sky blue?"))         # first time: the function runs
print(ask("Why is the sky blue?"))         # same prompt: read from disk, free and instant
print(ask.cache.stats().summary())
```

```
pretend this call cost you money: Why is the sky blue?
pretend this call cost you money: Why is the sky blue?
prompt-cache: 1 hit, 1 miss out of 2 lookups (50.0% hit rate)
  1 entry stored, 0.00 MB, 0 evictions, 1 call avoided
```

The cache survives the process, so running that script again is two hits and no calls at all. If you would rather hold the cache yourself:

```python
from prompt_cache import Cache

with Cache("~/.prompt_cache", ttl=86400, max_size_mb=200) as cache:
    answer = cache.get("Why is the sky blue?", model="gpt-4o")
    if answer is None:
        answer = call_the_model("Why is the sky blue?", model="gpt-4o")
        cache.set("Why is the sky blue?", answer, model="gpt-4o")
```

## What it does

- **Keys are hashes.** A key is the SHA-256 digest of the namespace, the normalised prompt and every parameter. A one-megabyte prompt and a three-word prompt are both 64 characters in the database, and no prompt is ever used as an index.
- **Parameters are part of the key.** `cache.get(p, model="gpt-4o")` and `cache.get(p, model="gpt-4o-mini")` are different entries, as are two different temperatures. Values that JSON does not know are keyed by type and `repr`, so `1`, `"1"` and `Decimal("1")` never collide.
- **Prompts are normalised first** (unless you pass `normalize=False`), so prompts that differ only in layout still hit. Exactly four steps, in order:
  1. `\r\n` and `\r` become `\n`;
  2. trailing spaces and tabs are stripped from the end of every line;
  3. a run of two or more blank lines collapses to one blank line;
  4. leading and trailing blank lines are dropped.

  Nothing else changes: case, inner spacing, punctuation and every non-ASCII character are kept exactly as written. `prompt_cache.normalize_prompt(text)` shows you the result.
- **Several processes can share one file.** Storage is SQLite in WAL mode with a 15 second busy timeout, so two workers writing at the same moment neither lose entries nor raise "database is locked".
- **A corrupt database is rebuilt, not raised.** If the file turns out not to be a database (truncated, half-written, or something else entirely), it is renamed to `cache.db.corrupt`, a fresh one takes its place, and a warning goes to the `prompt_cache` logger. Your caller sees a cache miss, never a crash.
- **Values round-trip faithfully.** JSON is used when the round trip keeps the same types, pickle otherwise, and the format is stored next to the blob, so a read never has to guess. A value that neither can store raises `TypeError` naming the type.
- **Entries expire and evict.** `ttl` is seconds; `prune()` removes what has expired. `max_size_mb` drops the least recently used entries when a write pushes the namespace over the limit.
- **`ttl=0` turns caching off** without changing any other code: nothing is written and every lookup misses. Useful for a `--no-cache` flag.
- **The cache directory is created on demand.** Building a `Cache` touches no files; the first read or write creates the folder and the database.

## API

```python
prompt_cache.cached(path=None, **kw) -> decorator
prompt_cache.Cache(path=None, *, ttl=None, max_size_mb=None, namespace=None, normalize=True)
prompt_cache.normalize_prompt(text) -> str
```

**`Cache(path=None, *, ttl=None, max_size_mb=None, namespace=None, normalize=True)`**

- `path` - a folder (the database goes in it as `cache.db`) or a `.db`/`.sqlite`/`.sqlite3` file. `None` means `~/.prompt_cache`. `":memory:"` gives a private cache that disappears with the process.
- `ttl` - seconds an entry stays valid. `None` means forever, `0` means never cache.
- `max_size_mb` - megabyte ceiling for this namespace; least recently used entries go first. A single value bigger than the ceiling is not stored at all (a warning is logged).
- `namespace` - the name of this logical cache, `"default"` if you do not pass one. **A cache only ever sees its own namespace**, so several of them can share one file without colliding, and `clear()`, `prune()`, `stats()` and eviction are all scoped to it.
- `normalize` - apply the normalisation above before hashing.

Methods:

- `.get(prompt, **params) -> value or None` - one indexed lookup; a miss writes nothing.
- `.set(prompt, value, **params) -> None` - store an answer.
- `.lookup(prompt, **params) -> (found, value)` - use this when `None` is a value you might have stored on purpose; `.get()` cannot tell that apart from a miss.
- `.wrap(func) -> callable` - the decorator form. The first argument is the prompt; every other argument is part of the key, and so are the function's module and qualified name, so two different functions never read each other's answers. An exception inside `func` is never cached. The wrapper exposes the cache as `wrapped.cache`.
- `.memoize(func)` - the same thing, when `@cache.memoize` reads better at the call site.
- `.stats() -> Stats`
- `.entries(limit=20) -> list of dict` - key, prompt preview (200 characters at most), size, timestamps, hit count.
- `.clear(namespace=None) -> int` - empty this namespace; `clear("other")` empties another one, `clear("*")` empties the whole file. Returns how many entries went.
- `.prune() -> int` - remove expired entries in this namespace.
- `.close()` - release the connection; safe to call twice, and the cache reopens by itself if used again.
- `.key(prompt, **params) -> str` - the digest an entry is stored under.
- Read-only properties: `.path`, `.ttl`, `.max_size_mb`, `.namespace`, `.normalize`, `.enabled`.
- Works as a context manager: `with Cache(...) as cache:`.

**`Stats(hits, misses, hit_rate, entries, size_mb, evictions, saved_calls)`** - a frozen dataclass with `.lookups`, `.summary()` (the two-line human text) and `.to_dict()` (JSON-safe). `hits`, `misses`, `evictions` and `saved_calls` count what *this* `Cache` object has seen since it was created; `entries` and `size_mb` are read from the database and describe everything in the namespace, whoever put it there. `saved_calls` is the subset of hits that came through `wrap()`/`memoize()`, that is, the calls that did not happen.

**`cached(path=None, **kw)`** - builds one `Cache` and hands it to every function it decorates. `cached(...)` takes the same keyword arguments as `Cache`; the decorated function exposes the cache as `func.cache`.

Errors are the plain built-ins: `TypeError` when a prompt is not a string or a value can be stored by neither JSON nor pickle (the message names the type), `ValueError` for a negative `ttl` or a `max_size_mb` of zero.

**A note on pickle.** Values that JSON cannot hold faithfully are pickled, and unpickling runs code. A cache file is local, private storage: treat it the way you treat any file your own program writes, and do not point a `Cache` at a database you did not create.

## CLI

```
prompt-cache                                        # what the default cache holds
prompt-cache --path ./.demo-cache stats             # ... for a different cache
prompt-cache list --limit 5                         # most recently used entries
prompt-cache list --json --output entries.json      # same thing as JSON
prompt-cache get "Why is the sky blue?" --param model=gpt-4o
prompt-cache set "Why is the sky blue?" "Rayleigh scattering." --param model=gpt-4o
prompt-cache prune                                  # drop expired entries
prompt-cache clear                                  # empty this namespace
prompt-cache clear --all                            # empty every namespace in the file
```

Every command takes `--path`, `--namespace` and `--ttl`. `get` exits 1 when there is nothing cached, so `prompt-cache get "..." >/dev/null || echo miss` works in a script. `--param KEY=VALUE` may be repeated; a value that parses as JSON becomes that JSON value (`0.7` is a number), anything else stays text (`gpt-4o` is a string). Output is UTF-8 whatever the console encoding is, so piping a cache full of Japanese or emoji prompts through another program is safe.

## License

MIT
