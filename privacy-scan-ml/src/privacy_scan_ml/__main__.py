"""Allow ``python -m privacy_scan_ml``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
