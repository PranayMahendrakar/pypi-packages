"""hallucination-check: check an answer against the sources it claims to use.

Quick use::

    import hallucination_check as hc
    report = hc.check(answer, sources)
    print(report.summary())
    print(report.unsupported)
"""
from ._core import Claim, GroundingChecker, GroundingReport, check, check_batch

__version__ = "0.1.0"

__all__ = [
    "Claim",
    "GroundingChecker",
    "GroundingReport",
    "check",
    "check_batch",
    "__version__",
]
