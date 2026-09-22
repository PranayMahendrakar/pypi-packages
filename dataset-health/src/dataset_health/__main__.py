"""Allow ``python -m dataset_health data.csv``."""

from .cli import main

raise SystemExit(main())
