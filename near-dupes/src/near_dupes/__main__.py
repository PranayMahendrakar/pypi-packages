"""Allow ``python -m near_dupes``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
