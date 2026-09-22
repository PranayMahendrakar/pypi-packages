"""Allow ``python -m semantic_dedup``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
