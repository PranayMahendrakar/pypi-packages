"""Run each queued package's tests under Python 3.11 + pandas 3, in isolation.

This machine has pandas 2. The release pipeline, and everyone who installs these
packages, has pandas 3. Code that writes into a `.to_numpy()` result works here and
raises "assignment destination is read-only" there, and a datetime built without an
explicit unit gets a different resolution on each. Both have already shipped bugs.

The first version of this script installed every package editable into ONE environment,
so they shadowed one another and the answers were noise: two packages came back broken
that are in fact fine. Each package is now installed, tested and uninstalled on its own,
and the tests run from a directory with no src/ beside them, so what is exercised is the
INSTALLED package rather than the source tree.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY3 = Path.home().parent / "PRECIS~1"  # placeholder, replaced below
PY3 = Path(r"C:\Users\PRECIS~1\AppData\Local\Temp\p3env\Scripts\python.exe")


def run(args, cwd=None):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, errors="replace")


def main() -> int:
    names = sys.argv[1:] or json.loads((ROOT / "approved.json").read_text(encoding="utf-8"))
    results: dict[str, str] = {}
    bad: list[str] = []

    for name in names:
        pkg = ROOT / name
        if not (pkg / "pyproject.toml").exists():
            results[name] = "no such package"
            continue

        install = run([str(PY3), "-m", "pip", "install", "-q", ".[dev]"], cwd=pkg)
        if install.returncode != 0:
            install = run([str(PY3), "-m", "pip", "install", "-q", "."], cwd=pkg)
        if install.returncode != 0:
            tail = (install.stderr or install.stdout).strip().splitlines()[-1:] or [""]
            results[name] = "INSTALL FAILED: " + tail[0][:80]
            bad.append(name)
            print("  %-24s INSTALL FAILED" % name, flush=True)
            continue

        work = Path(tempfile.mkdtemp())
        shutil.copytree(pkg / "tests", work / "tests")
        for extra in ("README.md", "pyproject.toml"):
            if (pkg / extra).exists():
                shutil.copy2(pkg / extra, work / extra)

        test = run(
            [str(PY3), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "tests"],
            cwd=work,
        )
        shutil.rmtree(work, ignore_errors=True)

        out = (test.stdout or "") + (test.stderr or "")
        lines = [line for line in out.strip().splitlines() if line.strip()]
        last = lines[-1] if lines else ""

        if test.returncode == 0:
            results[name] = "ok - " + last[:60]
            print("  %-24s ok     %s" % (name, last[:48]), flush=True)
        else:
            results[name] = "FAILS: " + last[:80]
            bad.append(name)
            print("  %-24s FAILS  %s" % (name, last[:48]), flush=True)

        run([str(PY3), "-m", "pip", "uninstall", "-y", "-q", name])

    print("\n  %d/%d clean on pandas 3" % (len(results) - len(bad), len(results)))
    if bad:
        print("  needs work: " + ", ".join(bad))
    (ROOT / "pandas3_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
