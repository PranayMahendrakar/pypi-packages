"""dataset-splitter: leakage-safe train/validation/test splits in one call."""
from ._io import load_table
from .groups import detect_id_columns
from .report import SplitReport
from .splitter import Split, Splitter, split

__version__ = "0.1.0"
__all__ = [
    "split",
    "Splitter",
    "Split",
    "SplitReport",
    "detect_id_columns",
    "load_table",
    "__version__",
]
