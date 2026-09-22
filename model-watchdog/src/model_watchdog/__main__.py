"""Allow ``python -m model_watchdog checkout-model``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
