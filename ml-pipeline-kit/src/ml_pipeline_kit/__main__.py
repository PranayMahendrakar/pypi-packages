"""Allow ``python -m ml_pipeline_kit``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
