"""Allow ``python -m text_quality_ai``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
