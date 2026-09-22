"""Allow ``python -m offline_ml``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
