"""auto-label: label text or tabular data from rules, a small model, and an optional LLM hook.

Quick use::

    import auto_label
    result = auto_label.label(texts, {"spam": ["free", "prize"], "work": ["meeting"]})
    result.to_frame()

For control, build a :class:`Labeler`, add rules, then call ``.label(data)``.
"""

from auto_label.labeler import Labeler, LabelResult, label
from auto_label.rules import Rule

__version__ = "0.1.0"

__all__ = ["Labeler", "LabelResult", "Rule", "label", "__version__"]
