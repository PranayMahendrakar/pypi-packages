"""The one-line entry point: :func:`detect_events`."""

from __future__ import annotations

import math
from typing import Any, Optional

from ._frames import iter_frames
from .detector import Detector
from .events import EventReport


def detect_events(frames: Any, *, fps: Optional[float] = None, **kw: Any) -> EventReport:
    """Run a :class:`Detector` over a whole sequence and report what it found.

    Args:
        frames: a list (or any iterable) of numpy arrays or PIL images, a numpy
            stack shaped ``(frames, height, width[, channels])``, a list of image
            paths, or a directory of numbered image files. Video files are not
            decoded; extract frames first (``ffmpeg -i clip.mp4 -vf fps=10
            frames/%05d.png``).
        fps: frames per second. With it every time in the report is in
            seconds; without it, times are frame numbers.
        **kw: passed to :class:`Detector` (``sensitivity``, ``background_frames``,
            ``dwell``, ``hold``). ``dwell`` and ``hold`` are on the same clock
            as the report: seconds when ``fps`` is given.

    Returns:
        An :class:`EventReport` with ``.events``, ``.by_kind``, ``.timeline``,
        ``.summary()`` and ``.to_dict()``.

    Raises:
        ValueError: unreadable input, or a bad ``fps`` or detector setting.
        FileNotFoundError: ``frames`` names a path that does not exist.
    """
    if fps is not None:
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("fps must be a positive number or None; got {!r}".format(fps))
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a positive number; got {}".format(fps))
    detector = Detector(**kw)
    pairs, source, _ = iter_frames(frames)
    detector._source = source
    detector._fps = fps
    for index, (frame, name) in enumerate(pairs):
        timestamp = index / fps if fps else None
        detector._update(frame, timestamp, name)
    return detector.report()
