"""Allow ``python -m image_dedup_ai``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
