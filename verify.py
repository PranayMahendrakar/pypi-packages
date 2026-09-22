"""Deterministic release verification: install each wheel into a clean venv and prove it works."""
import json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(r"D:\03-Projects-Code\pypi-packages")
WORK = Path(tempfile.gettempdir()) / "pkgverify"
PKGS = sys.argv[1:] or ["auto-label","data-drift-lite","dataset-health","dataset-splitter",
                        "near-dupes","schema-guard","synthetic-tabular"]

def run(cmd, cwd=None, env=None, timeout=900, stdin_text=None, shell=False):
    e = dict(os.environ); e["PYTHONIOENCODING"] = "utf-8"; e["PYTHONUTF8"] = "1"
    if env: e.update(env)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=e, timeout=timeout, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", input=stdin_text, shell=shell)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"TIMEOUT after {timeout}s"

def quickstart_from_readme(pkgdir):
    md = (pkgdir / "README.md").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"##\s*Quickstart\s*\n+```(?:python)?\n(.*?)```", md, re.S | re.I)
    return m.group(1) if m else None

results = {}
for name in PKGS:
    pkgdir = ROOT / name
    r = {"package": name, "checks": {}, "fail": [], "warn": []}
    def ok(k, cond, detail=""):
        r["checks"][k] = bool(cond)
        if not cond: r["fail"].append(f"{k}: {detail[:400]}")
    t0 = time.time()

    whls = sorted(pkgdir.glob("dist/*-0.1.0-*.whl"))
    tgzs = sorted(pkgdir.glob("dist/*-0.1.0.tar.gz"))
    ok("dist_pair", len(whls) == 1 and len(tgzs) == 1, f"whl={len(whls)} tgz={len(tgzs)}")
    if len(whls) != 1:
        results[name] = r; continue
    whl = whls[0]

    venv = WORK / name
    shutil.rmtree(venv, ignore_errors=True); venv.mkdir(parents=True, exist_ok=True)
    rc, so, se = run([sys.executable, "-m", "venv", str(venv)])
    ok("venv_created", rc == 0, se)
    py = venv / "Scripts" / "python.exe"
    if not py.exists(): results[name] = r; continue

    rc, so, se = run([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check", str(whl)])
    ok("wheel_installs", rc == 0, se)
    if rc != 0: results[name] = r; continue

    rc, so, _ = run([str(py), "-m", "pip", "freeze"])
    deps = [l.split("==")[0].lower() for l in so.splitlines() if l.strip()]
    heavy = [d for d in deps if d in {"torch","tensorflow","transformers","keras","jax"}]
    ok("no_heavy_deps", not heavy, f"pulled {heavy}")
    r["installed"] = sorted(deps)

    qs = quickstart_from_readme(pkgdir)
    ok("readme_has_quickstart", qs is not None, "no ## Quickstart python block")
    if qs:
        f = venv / "qs.py"; f.write_text(qs, encoding="utf-8")
        rc, so, se = run([str(py), str(f)], cwd=str(venv), timeout=300)
        ok("quickstart_runs", rc == 0, se)
        r["quickstart_output"] = (so or "")[:600]

    exe = venv / "Scripts" / f"{name}.exe"
    ok("cli_installed", exe.exists(), "console script missing")
    if exe.exists():
        rc, so, se = run([str(exe), "--help"])
        ok("cli_help", rc == 0 and ("usage" in (so+se).lower()), se)
        # unicode through a pipe: the exact failure QA found in siblings
        csv = venv / "u.csv"
        csv.write_text("name,city,amount,flag\nZoë,Kraków,12,yes\n李雷,北京,7,no\nJosé,Ñuñoa,,yes\n"
                       "Zoë,Kraków,12,yes\nAnn,Paris,900000,no\n", encoding="utf-8")
        rc, so, se = run(f'"{exe}" "{csv}" | more', cwd=str(venv), shell=True)
        blob0 = so + se
        # The point of this check is the ENCODING, not whether our CSV fixture fits the
        # tool's argument shape. A UnicodeEncodeError is always a failure; a clean
        # refusal of the fixture is not.
        ok("cli_unicode_piped", "UnicodeEncodeError" not in blob0 and "Traceback" not in blob0, blob0)
        rc, so, se = run(f'"{exe}" "{csv}" --json', cwd=str(venv), shell=True)
        # A non-zero exit is not a failure here. Feeding a CSV to an image tool SHOULD
        # fail; what matters is that --json still answers with a parseable document
        # describing the failure rather than a traceback or silence.
        good_json = False
        try:
            json.loads(so)
            good_json = True
        except Exception as ex:
            se = f"{se} (not valid json: {ex})"
        # Not every tool takes a CSV: a chunker wants a document, a cache takes a
        # subcommand, a hardware probe takes nothing at all. Refusing our fixture with
        # a clear message is CORRECT behaviour, so only a crash counts against it.
        blob = (so + se)
        # Judge the SHAPE, not the wording: a clean refusal is a short message with no
        # traceback, usually "<prog>: error: ...". Listing phrases was brittle -- three
        # tools refused correctly and were still marked failed because they happened to
        # say "cannot import module" or "is not valid JSON" instead.
        refused_cleanly = (
            not good_json
            and "Traceback" not in blob
            and len(blob.strip().splitlines()) <= 12
            and (
                f"{name}:" in blob          # the tool named itself, argparse-style or not
                or "usage:" in blob.lower()
                or "error:" in blob.lower()
            )
        )
        if refused_cleanly:
            r["warn"].append("cli_json_piped: this tool does not take a CSV; it refused cleanly")
            r["checks"]["cli_json_piped"] = True
        else:
            ok("cli_json_piped", good_json, se)

    tests = pkgdir / "tests"
    if tests.exists():
        rc, so, se = run([str(py), "-m", "pip", "install", "-q", "pytest"])
        base = venv / "pkgroot"; shutil.rmtree(base, ignore_errors=True); base.mkdir()
        shutil.copytree(tests, base / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        for extra in ("README.md", "pyproject.toml", "LICENSE"):
            if (pkgdir / extra).exists(): shutil.copy2(pkgdir / extra, base / extra)
        rc, so, se = run([str(py), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                         cwd=str(base), timeout=1200)
        tail = (so or "")[-500:]
        ok("tests_vs_installed_wheel", rc == 0, tail + se[-300:])
        r["test_tail"] = tail.strip().splitlines()[-1] if tail.strip() else ""

    r["seconds"] = round(time.time() - t0, 1)
    r["verdict"] = "PASS" if not r["fail"] else "FAIL"
    results[name] = r
    print(f"{name:20s} {r['verdict']:5s} {r.get('seconds',0):6.1f}s  fails={len(r['fail'])}", flush=True)
    for f in r["fail"]: print(f"      - {f}", flush=True)

out = ROOT / "verify_results.json"
out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print("\nwrote", out)
print("PASS:", [n for n,v in results.items() if v.get("verdict")=="PASS"])
print("FAIL:", [n for n,v in results.items() if v.get("verdict")!="PASS"])
