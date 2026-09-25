"""Count objects in images or across video frames.

Bring a detector for real objects, or use the built-in classical counter, which
counts high-contrast blobs against a plain background (parts on a conveyor,
cells on a slide, bolts on a tray) and knows nothing about what an object is.

    result = object_counter_ai.count(image)                  # classical blobs
    result = object_counter_ai.count(image, detector=model)  # your boxes
    counter = object_counter_ai.Counter().line((0, 50), (200, 50))
"""
from __future__ import annotations

import logging

from ._core import count
from ._counter import Counter
from ._result import CountResult

__version__ = "0.1.0"

__all__ = ["count", "Counter", "CountResult", "__version__"]

logging.getLogger(__name__).addHandler(logging.NullHandler())
