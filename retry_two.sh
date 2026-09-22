#!/usr/bin/env bash
# Wait out the PyPI new-project rate limit, then upload the two remaining packages.
ROOT="/d/03-Projects-Code/pypi-packages"
for attempt in 1 2 3 4 5 6; do
  sleep 900   # 15 min between attempts
  for pkg in auto-label schema-guard; do
    code=$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$pkg/json")
    [ "$code" = "200" ] && continue   # already live
    echo "[attempt $attempt] uploading $pkg"
    out=$(python -m twine upload --non-interactive "$ROOT/$pkg"/dist/* 2>&1)
    if echo "$out" | grep -q "429"; then
      echo "[attempt $attempt] $pkg: still rate limited"
    elif echo "$out" | grep -qiE "error|failed"; then
      echo "[attempt $attempt] $pkg: FAILED"; echo "$out" | grep -iE "error|reason" | head -3
    else
      echo "[attempt $attempt] $pkg: uploaded"
    fi
    sleep 60
  done
  a=$(curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/auto-label/json)
  s=$(curl -s -o /dev/null -w '%{http_code}' https://pypi.org/pypi/schema-guard/json)
  if [ "$a" = "200" ] && [ "$s" = "200" ]; then
    echo "BOTH LIVE: https://pypi.org/project/auto-label/ and https://pypi.org/project/schema-guard/"
    exit 0
  fi
done
echo "gave up after 6 attempts; auto-label=$a schema-guard=$s"
exit 1
