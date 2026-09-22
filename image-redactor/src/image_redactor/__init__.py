"""image-redactor: blur or mask faces, plates and regions you mark.

Quick use::

    import image_redactor
    result = image_redactor.redact(photo, regions=[(40, 20, 100, 90)], method="pixelate")
    print(result.summary())
    result.save("safe.png")

Plug in a real detector - this package deliberately depends on none::

    image_redactor.redact(photo, detector=my_face_model.predict)

The built-in ``detect_faces`` and ``detect_plates`` are weak pixel heuristics,
not models. They are for hiding obvious cases and for exercising a pipeline.
Do not rely on them for privacy compliance.
"""
from ._core import (
    BUILTIN_DETECTORS,
    HEURISTIC_CAVEAT,
    IRREVERSIBLE_METHODS,
    METHOD_HELP,
    METHODS,
    DetectorReport,
    RedactResult,
    Redactor,
    detect_faces,
    detect_plates,
    redact,
    redact_file,
)

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_DETECTORS",
    "DetectorReport",
    "HEURISTIC_CAVEAT",
    "IRREVERSIBLE_METHODS",
    "METHODS",
    "METHOD_HELP",
    "RedactResult",
    "Redactor",
    "detect_faces",
    "detect_plates",
    "redact",
    "redact_file",
    "__version__",
]
