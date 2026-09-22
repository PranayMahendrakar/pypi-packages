"""Allow ``python -m rag_chunker``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
