"""image-dedup-ai: find duplicate and near-duplicate images across a large folder.

Perceptual hashes (pHash by default) are a heuristic: they find resized and
re-compressed copies, not rotated, flipped or heavily cropped ones. Hashes are
kept in a SQLite index keyed on path, size and modification time, so a folder
is hashed once and re-checked instantly afterwards.

    import image_dedup_ai
    result = image_dedup_ai.find_duplicates("photos/")
    print(result.summary())
"""
from ._hashing import FLAT_STD, IMAGE_EXTENSIONS, METHODS, hash_image
from ._index import Index, find_duplicates
from ._result import AddReport, DedupeResult, Match, Member

__version__ = "0.1.0"

__all__ = [
    "Index",
    "find_duplicates",
    "hash_image",
    "DedupeResult",
    "Match",
    "Member",
    "AddReport",
    "METHODS",
    "IMAGE_EXTENSIONS",
    "FLAT_STD",
    "__version__",
]
