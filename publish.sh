#!/usr/bin/env bash
# Upload approved packages to PyPI, one at a time, verifying each before and after.
# Usage: bash publish.sh <package-name> [<package-name> ...]
# Credentials come from ~/.pypirc. Nothing is uploaded unless every gate passes.
set -u
ROOT="/d/03-Projects-Code/pypi-packages"
PAUSE=45
ok=(); skipped=(); failed=()

for pkg in "$@"; do
  d="$ROOT/$pkg"
  echo "=============================================================="
  echo "  $pkg"
  echo "=============================================================="

  if [ ! -d "$d" ]; then echo "  SKIP: no folder"; skipped+=("$pkg (no folder)"); continue; fi

  whl=$(ls "$d"/dist/*-0.1.0-*.whl 2>/dev/null | wc -l)
  tgz=$(ls "$d"/dist/*-0.1.0.tar.gz 2>/dev/null | wc -l)
  if [ "$whl" != "1" ] || [ "$tgz" != "1" ]; then
    echo "  SKIP: expected exactly 1 wheel + 1 sdist for 0.1.0, found $whl + $tgz"
    skipped+=("$pkg (bad dist)"); continue
  fi

  if curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$pkg/json" | grep -q 200; then
    echo "  SKIP: name already exists on PyPI"
    skipped+=("$pkg (already on pypi)"); continue
  fi

  echo "  -- twine check --"
  if ! python -m twine check "$d"/dist/*; then
    echo "  SKIP: twine check failed"; skipped+=("$pkg (twine check)"); continue
  fi

  echo "  -- uploading --"
  out=$(python -m twine upload --non-interactive "$d"/dist/* 2>&1)
  if echo "$out" | grep -qi "too similar to an existing"; then
    echo "  BLOCKED: PyPI says this name is too close to an existing project."
    echo "           Rename it in pyproject.toml and rebuild - retrying will never work."
    failed+=("$pkg (name too similar)")
    continue
  fi
  if echo "$out" | grep -q "429"; then
    echo "  rate limited - stopping for now, the window refills in about a day"
    failed+=("$pkg (rate limited)")
    break
  fi
  if ! echo "$out" | grep -qiE "error|failed"; then
    sleep 10
    code=$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$pkg/json")
    if [ "$code" = "200" ]; then
      echo "  LIVE: https://pypi.org/project/$pkg/"
      ok+=("$pkg")
    else
      echo "  WARN: upload reported success but PyPI returns $code (index lag?)"
      ok+=("$pkg (unconfirmed)")
    fi
  else
    echo "  FAILED to upload"; failed+=("$pkg")
  fi

  echo "  pausing ${PAUSE}s"; sleep "$PAUSE"
done

echo
echo "=============================== RESULT ==============================="
printf 'published : %s\n' "${ok[*]:-none}"
printf 'skipped   : %s\n' "${skipped[*]:-none}"
printf 'failed    : %s\n' "${failed[*]:-none}"
