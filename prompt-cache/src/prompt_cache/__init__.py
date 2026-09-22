"""prompt-cache: cache LLM answers on disk so repeated prompts cost nothing.

Three lines get you the whole thing::

    import prompt_cache

    @prompt_cache.cached()
    def ask(prompt, model="gpt-4o"):
        return call_the_model(prompt, model=model)

The second time ``ask`` sees a prompt it returns the stored answer instead of
calling the model. Storage is a single SQLite file, so several processes can
share one cache safely, and nothing outside the standard library is needed.
"""

from ._keys import make_key, normalize_prompt
from ._serialize import decode, encode
from ._store import Store
from .cache import (
    BUSY_TIMEOUT,
    DB_FILENAME,
    DEFAULT_DIR,
    DEFAULT_NAMESPACE,
    Cache,
    cached,
    resolve_path,
)
from .stats import Stats

__version__ = "0.1.0"

__all__ = [
    "BUSY_TIMEOUT",
    "DB_FILENAME",
    "DEFAULT_DIR",
    "DEFAULT_NAMESPACE",
    "Cache",
    "Stats",
    "Store",
    "__version__",
    "cached",
    "decode",
    "encode",
    "make_key",
    "normalize_prompt",
    "resolve_path",
]
