"""Allow ``python -m ml_inference_profiler``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
