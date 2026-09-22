"""Allow ``python -m multilingual_text``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
