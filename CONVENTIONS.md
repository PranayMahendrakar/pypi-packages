# Package family conventions (read fully before writing any package)

Every package in this folder is one member of a family. They must feel identical in
shape, so a user who learns one can use all of them.

## Location and layout
- One folder per distribution: `D:\03-Projects-Code\pypi-packages\<dist-name>\`
- src layout:
  ```
  <dist-name>/
    pyproject.toml
    README.md
    LICENSE                  (MIT, copyright 2026 Pranay Mahendrakar)
    src/<import_name>/__init__.py   (public API re-exported here, plus __version__)
    src/<import_name>/cli.py        (argparse only, `main()` entry point)
    src/<import_name>/...           (implementation modules)
    tests/test_*.py                 (pytest)
  ```
- `import_name` is the dist name with hyphens turned into underscores.

## pyproject.toml (hatchling)
```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "<dist-name>"
version = "0.1.0"
description = "<one line, what it solves, no buzzwords>"
readme = "README.md"
requires-python = ">=3.9"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Pranay Mahendrakar" }]
keywords = [...]
classifiers = [
  "Development Status :: 4 - Beta",
  "Intended Audience :: Developers",
  "Intended Audience :: Science/Research",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3 :: Only",
  "Operating System :: OS Independent",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [ ... minimal ... ]

[project.optional-dependencies]
# heavy or niche deps go here, never in `dependencies`
parquet = ["pyarrow>=12"]
dev = ["pytest>=7"]

[project.scripts]
<dist-name> = "<import_name>.cli:main"

[project.urls]
Homepage = "https://pypi.org/project/<dist-name>/"
Author = "https://pypi.org/user/pranaymahendrakar/"

[tool.hatch.build.targets.wheel]
packages = ["src/<import_name>"]
```

## API shape ("easy" is the product)
- One top-level convenience function for the 90% case, callable in one line,
  e.g. `report = dataset_doctor.diagnose(df)` or `clean_df, log = smart_clean.clean(df)`.
- One class underneath for people who need control, with sensible defaults.
- Accept a pandas DataFrame OR a path to `.csv` / `.parquet` wherever input is tabular.
- Results are plain dataclasses or dicts with a `.summary()` (human text) and
  `.to_dict()` (JSON-safe). No custom exceptions unless truly needed.
- Deterministic: any randomness takes `random_state`.
- Typed signatures and short docstrings on every public function.
- Library code never prints. Use `logging.getLogger(__name__)`.
- Must run on Windows, Linux, macOS. No shell calls, no hard-coded paths.
- Pure Python + numpy/pandas/scikit-learn wherever possible. Do not pull in
  torch, transformers, or any model download for the default install.

## CLI
- `<dist-name> --help` works. Typical: `<dist-name> data.csv` prints the summary,
  `--json` prints `to_dict()` as JSON, `--output path` writes results.

## README.md (exact section order)
1. `# <dist-name>` and one sentence on the problem it solves.
2. `## Install` with `pip install <dist-name>`.
3. `## Quickstart` with a 3-6 line, copy-pasteable example that works on a tiny
   inline DataFrame (no external files). This exact block is what QA runs.
4. `## What it checks` / `## What it does` bullets.
5. `## API` short reference of the public functions and the result object.
6. `## CLI` usage.
7. `## License` MIT.

## Tests
- pytest, no network, no model downloads, whole suite under 15 seconds.
- Cover: the quickstart path, an empty DataFrame, a single row, an all-NaN column,
  mixed dtypes, unicode text, and every public function at least once.
- Tests must pass with `python -m pytest -q` from the package folder.

## Build and proof (every package, no exceptions)
1. `python -m build` from the package folder produces `dist/*.whl` and `dist/*.tar.gz`.
2. Fresh-venv proof, run from a scratch dir OUTSIDE the package folder:
   `python -m venv .venv-proof && .venv-proof/Scripts/python -m pip install <path-to-wheel>`
   then run the README quickstart verbatim as a script in that venv, and run
   `<dist-name> --help`. Both must succeed. Delete the proof venv afterwards.
3. `python -m twine check dist/*` passes.

## Hard rules
- NEVER run `twine upload` or anything that publishes. Publishing is done by the
  owner after review.
- Do not create git repositories inside package folders.
- Do not modify any other package's folder.
- Version stays 0.1.0 for first release.

## Windows console and encoding safety (added after QA round 1 found this in 4 of 5 packages)
- At the top of `cli.main()`, make stdout/stderr UTF-8 tolerant so non-Latin text never raises
  UnicodeEncodeError under pipes, subprocess capture or CI logs:
  ```python
  for stream in (sys.stdout, sys.stderr):
      if hasattr(stream, "reconfigure"):
          stream.reconfigure(encoding="utf-8", errors="replace")
  ```
- Every file opened for writing uses `encoding="utf-8"`; JSON is written with `ensure_ascii=False`.
- `summary()` text uses plain ASCII punctuation (no arrows, bullets or box characters).
- A DataFrame with duplicate column names must raise a clear `ValueError` naming the duplicates
  at the entry point, never an AttributeError from inside pandas.
- Any "auto" heuristic (auto group detection, auto target type, auto kind) must never silently
  produce an empty or degenerate result: detect it, fall back, and record a warning in the report.


## pandas 2 and pandas 3 (added after four packages broke on 3 while passing on 2)

This machine has pandas 2. The release pipeline and every user installing today have
pandas 3. Code can pass every test here and be broken for everyone. Check any package
that touches pandas against the pandas 3 environment before approving it:

    python check_pandas3.py <name>

Four traps found so far, all silent on pandas 2:

- **`.to_numpy()` returns a READ-ONLY array** under copy-on-write, the pandas 3 default.
  Writing into it raises "assignment destination is read-only". Pass `copy=True` at every
  call site you write into - timeseries-anomaly failed 52 of its own tests this way.
- **Datetime resolution is no longer nanoseconds.** `pd.DatetimeTZDtype(tz=...)` and
  `pd.to_datetime` take their unit from the pandas default, so one schema produced
  columns that would not concat. Pin the unit if you care that it is stable.
- **Retired offset aliases.** `'H'`, `'T'`, `'S'` warned on pandas 2 and are gone in 3.
  Translate them rather than passing them through, or user code that worked for years
  starts raising.
- **`pd.Timedelta(offset)` refuses a Day offset** on pandas 3. Use `offset.nanos`, which
  answers "is this a fixed duration, and how long" correctly on both, and still refuses
  weeks and months, which genuinely are not fixed.

And in tests: never assert on a dtype's spelling. A string column is `object` on pandas 2
and `str` on pandas 3; a timedelta is `[ns]` then `[s]`. Assert the behaviour instead.
