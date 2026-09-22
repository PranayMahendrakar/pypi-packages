"""smartclean-df: turn a messy table into a tidy one in one call, and say what changed."""
from .core import FORWARD_FILL, Cleaner, clean
from .result import Action, CleanResult

__version__ = "0.1.0"
__all__ = ["clean", "Cleaner", "CleanResult", "Action", "FORWARD_FILL", "__version__"]
