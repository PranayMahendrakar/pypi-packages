"""The :func:`guard` decorator: enforce a schema on the first DataFrame argument of a function."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import pandas as pd

from .schema import as_schema

F = TypeVar("F", bound=Callable[..., Any])


def guard(schema: Any, mode: str = "coerce", *, extra: str = "drop", missing: str = "fill") -> Callable[[F], F]:
    """Decorator that runs :meth:`Schema.enforce` on the first DataFrame argument.

    ``schema`` may be a :class:`Schema`, a schema dict or a path to a saved schema JSON.
    The first positional argument that is a DataFrame is replaced by the enforced frame
    (keyword arguments are searched when no positional one is a DataFrame), so methods
    work too: ``self`` is skipped naturally. ``mode``, ``extra`` and ``missing`` are
    passed straight to :meth:`Schema.enforce`; with ``mode="strict"`` the function is
    never called when the frame does not match and :class:`SchemaError` is raised instead.
    """
    resolved = as_schema(schema)
    if mode not in ("coerce", "strict"):
        raise ValueError(f"mode must be 'coerce' or 'strict', got {mode!r}")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for position, value in enumerate(args):
                if isinstance(value, pd.DataFrame):
                    fixed = resolved.enforce(value, mode=mode, extra=extra, missing=missing)
                    new_args = args[:position] + (fixed,) + args[position + 1 :]
                    return func(*new_args, **kwargs)
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    fixed = resolved.enforce(value, mode=mode, extra=extra, missing=missing)
                    new_kwargs = dict(kwargs)
                    new_kwargs[key] = fixed
                    return func(*args, **new_kwargs)
            raise TypeError(
                f"{func.__name__}() was called without a DataFrame argument; "
                "@guard needs one to enforce the schema on"
            )

        wrapper.schema = resolved  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
