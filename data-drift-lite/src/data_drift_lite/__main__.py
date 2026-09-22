"""Allow ``python -m data_drift_lite reference.csv current.csv``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
