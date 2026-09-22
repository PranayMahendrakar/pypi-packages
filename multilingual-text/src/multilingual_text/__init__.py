"""multilingual-text: detect, normalise and transliterate text with one API.

Quick use::

    import multilingual_text as mlt

    d = mlt.detect("El perro es muy grande")
    d.language, d.confidence, d.reliable      # "es", 0.93, True
    mlt.normalize("  “smart” quotes — and hard spaces ")
    mlt.transliterate("नमस्ते")                  # "namaste"
    mlt.script_of("مرحبا"), mlt.is_rtl("مرحبا")  # "Arabic", True

Every result object explains itself: ``d.summary()`` is one line of plain
ASCII, ``d.to_dict()`` is JSON-safe.  Nothing is downloaded, nothing is
trained, and the standard library is the only dependency.
"""
from ._detect import (
    UNDETERMINED,
    Detection,
    char_script_of,
    detect,
    detect_batch,
    is_rtl,
    language_name,
    script_of,
)
from ._normalize import CASE_MODES, FORMS, normalize
from ._scripts import RTL_SCRIPTS, script_names
from ._translit import SUPPORTED_SCRIPTS, TARGETS, Transliteration, transliterate
from ._words import LANGUAGE_NAMES, SUPPORTED

__version__ = "0.1.0"

#: Every ISO 639-1 code :func:`detect` can return, apart from ``"und"``.
SUPPORTED_LANGUAGES = SUPPORTED

__all__ = [
    "CASE_MODES",
    "Detection",
    "FORMS",
    "LANGUAGE_NAMES",
    "RTL_SCRIPTS",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_SCRIPTS",
    "TARGETS",
    "Transliteration",
    "UNDETERMINED",
    "__version__",
    "char_script_of",
    "detect",
    "detect_batch",
    "is_rtl",
    "language_name",
    "normalize",
    "script_names",
    "script_of",
    "transliterate",
]
