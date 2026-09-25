"""``python -m speaker_diarize_lite`` runs the command line tool."""

import sys

from .cli import main

sys.exit(main())
