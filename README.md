# pypi-packages

A suite of small, single-purpose Python libraries for data quality, machine-learning
operations and signal analysis. Each directory is an independent package with its own
tests, documentation and command line interface, published separately to PyPI.

The packages share no code. They live in one repository so they can share one release
workflow and one set of conventions, not because they are coupled.

## Conventions

`CONVENTIONS.md` is the contract every package follows: src layout, hatchling build,
a README whose Quickstart block is executable and is what the tests run, a command line
interface that survives being piped, and no heavyweight dependencies in a default
install.

## Releasing

Releases go out through GitHub Actions using PyPI trusted publishing, so no API token is
stored anywhere. `.github/workflows/publish.yml` builds and publishes one package at a
time, chosen when the workflow is run.

Each package name needs a trusted publisher registered on PyPI pointing at this
repository and that workflow file. See `TRUSTED-PUBLISHING.md`.

## What gets released

Nothing is released because it builds. A package reaches the release queue only after
`verify.py` passes against the INSTALLED wheel rather than the source tree: a clean
virtual environment, the README quickstart run verbatim, the command line interface
exercised and piped with non-ASCII text, and the package's own test suite run against
what a user would actually install. `APPROVED.md` records what was fixed and why, and
what is deliberately held back.
