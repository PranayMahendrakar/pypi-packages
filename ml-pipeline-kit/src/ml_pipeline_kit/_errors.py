"""The three errors this package raises, all subclasses of one base."""
from __future__ import annotations

from typing import Optional


class PipelineError(RuntimeError):
    """Base class for every error raised by ml-pipeline-kit."""


class StepError(PipelineError):
    """A step raised, and the pipeline was told to let it through.

    The message names the pipeline, the step and how many rows reached it; the
    original exception is kept as ``__cause__``, so ``raise ... from`` chaining
    shows both without anyone having to read a bare traceback.
    """

    def __init__(self, message: str, *, step: str, rows_in: Optional[int] = None) -> None:
        super().__init__(message)
        self.step = step
        self.rows_in = rows_in


class ValidationError(PipelineError):
    """A check failed while the pipeline was called as a function.

    ``Pipeline.run()`` records failed checks in the result instead of raising;
    only ``Pipeline.__call__`` raises this, because it has nothing but the
    output value to hand back and must not hand back a half-finished one.
    """

    def __init__(self, message: str, *, check: str) -> None:
        super().__init__(message)
        self.check = check
