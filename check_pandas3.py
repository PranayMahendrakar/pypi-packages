"""Run each approved package's own tests under Python 3.11 + pandas 3.

This machine has pandas 2; the release pipeline and every real user have pandas 3.
Two packages have already shipped-or-nearly-shipped with faults that are invisible
here and fatal there, so the queue gets checked before the publisher meets it.
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(r"D:\03-Projects-Code\pypi-packages")
PY3 = Path(r"C:\Users\PRECIS~1\AppData\Local\Temp\p3env\Scripts\python.exe")

names = sys.argv[1:] or json.loads((ROOT / "approved.json").read_text(encoding="utf-8"))
results = {}
for name in names:
    pkg = ROOT / name
    if not (pkg / "pyproject.toml").exists():
        results[name] = ("SKIP", "no package folder")
        continue
    install = subprocess.run([str(PY3), "-m", "pip", "install", "-q", "-e", "."],
                             cwd=pkg, capture_output=True, text=True, errors="replace")
    if install.returncode != 0:
        tail = (install.stderr or install.stdout).strip().splitlines()[-1:] or [""]
        results[name] = ("INSTALL FAIL", tail[0][:90])
        print(f"  {name:24s} INSTALL FAIL  {tail[0][:70]}", flush=True)
        continue
    run = subprocess.run([str(PY3), "-m", "pytest", "-q", "--no-header", "-x", "-q"],
                         cwd=pkg, capture_output=True, text=True, errors="replace")
    out = (run.stdout or "") + (run.stderr or "")
    last = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
    verdict = "ok" if run.returncode == 0 else "FAILS ON PANDAS 3"
    results[name] = (verdict, last[0][:90])
    print(f"  {name:24s} {verdict:18s} {last[0][:60]}", flush=True)

bad = [n for n, (v, _) in results.items() if v != "ok"]
print(f"\n  {len(results) - len(bad)}/{len(results)} clean on pandas 3")
if bad:
    print("  needs work:", ", ".join(bad))
(ROOT / "pandas3_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
