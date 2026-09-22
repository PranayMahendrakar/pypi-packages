"""Allow ``python -m prompt_cache``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
