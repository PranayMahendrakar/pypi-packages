<#
.SYNOPSIS
  Publishes up to N verified packages to PyPI per run. Intended for a daily 17:00 scheduled task.
.DESCRIPTION
  Safety model: this script NEVER decides what is good. It publishes only names listed in
  approved.json, which is written by hand after a package has passed verification. Anything
  not in that file is ignored even if it has a perfectly good wheel sitting in dist/.
  It stops for the day on the first HTTP 429 so the account is not pushed further into
  PyPI's new-project rate limit.
#>
[CmdletBinding()]
param(
    [int]$Count = 5,
    [string]$Root = "D:\03-Projects-Code\pypi-packages",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$approvedPath = Join-Path $Root "approved.json"
$logPath      = Join-Path $Root "publish-log.txt"
$statePath    = Join-Path $Root "published.json"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $line = "{0} [{1}] {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Level, $Message
    Write-Output $line
    Add-Content -Path $logPath -Value $line -Encoding utf8
}

function Test-OnPyPI {
    param([string]$Name)
    try {
        $r = Invoke-WebRequest -Uri "https://pypi.org/pypi/$Name/json" -Method Head `
                               -UseBasicParsing -TimeoutSec 30
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

Write-Log "=== run start (Count=$Count DryRun=$DryRun) ==="

if (-not (Test-Path $approvedPath)) {
    Write-Log "no approved.json at $approvedPath - nothing is cleared for release" "WARN"
    exit 0
}

$approved = (Get-Content $approvedPath -Raw -Encoding utf8 | ConvertFrom-Json)
if ($null -eq $approved -or $approved.Count -eq 0) {
    Write-Log "approved.json is empty - nothing to publish" "WARN"
    exit 0
}

# Skip anything already on PyPI so a rerun is always safe.
$queue = @()
foreach ($name in $approved) {
    if (Test-OnPyPI -Name $name) {
        Write-Log "$name already on PyPI - skipping"
    } else {
        $queue += $name
    }
}

if ($queue.Count -eq 0) {
    Write-Log "every approved package is already published - done"
    exit 0
}

$batch = $queue | Select-Object -First $Count
Write-Log ("today's batch ({0} of {1} remaining): {2}" -f $batch.Count, $queue.Count, ($batch -join ", "))

$published = @()
$failed    = @()

foreach ($name in $batch) {
    $dir = Join-Path $Root $name
    Write-Log "--- $name ---"

    if (-not (Test-Path $dir)) {
        Write-Log "$name : no package folder, skipping" "ERROR"
        $failed += $name
        continue
    }

    $dist = Join-Path $dir "dist"
    $whl = @(Get-ChildItem -Path $dist -Filter "*.whl" -ErrorAction SilentlyContinue)
    $tgz = @(Get-ChildItem -Path $dist -Filter "*.tar.gz" -ErrorAction SilentlyContinue)
    if ($whl.Count -ne 1 -or $tgz.Count -ne 1) {
        Write-Log ("{0} : expected 1 wheel + 1 sdist, found {1} + {2}" -f $name, $whl.Count, $tgz.Count) "ERROR"
        $failed += $name
        continue
    }

    # Refuse to ship a wheel that is older than the source it came from.
    $newestSource = Get-ChildItem -Path (Join-Path $dir "src") -Recurse -Filter "*.py" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -ne $newestSource -and $newestSource.LastWriteTime -gt $whl[0].LastWriteTime) {
        Write-Log "$name : source is newer than the built wheel - needs a rebuild, skipping" "ERROR"
        $failed += $name
        continue
    }

    Write-Log "$name : twine check"
    $check = & python -m twine check "$dist\*" 2>&1 | Out-String
    if ($check -notmatch "PASSED" -or $check -match "FAILED") {
        Write-Log "$name : twine check did not pass, skipping" "ERROR"
        Write-Log $check.Trim() "ERROR"
        $failed += $name
        continue
    }

    if ($DryRun) {
        Write-Log "$name : DRY RUN - would upload now (not recorded as published)"
        continue
    }

    Write-Log "$name : uploading"
    $out = & python -m twine upload --non-interactive "$dist\*" 2>&1 | Out-String

    if ($out -match "429") {
        Write-Log "$name : HTTP 429 rate limit - stopping for today, will resume next run" "WARN"
        break
    }
    if ($out -match "400|403|ERROR|HTTPError") {
        Write-Log "$name : upload failed" "ERROR"
        Write-Log ($out.Trim()) "ERROR"
        $failed += $name
        Start-Sleep -Seconds 60
        continue
    }

    Start-Sleep -Seconds 15
    if (Test-OnPyPI -Name $name) {
        Write-Log "$name : LIVE at https://pypi.org/project/$name/"
    } else {
        Write-Log "$name : upload reported success but index has not caught up yet" "WARN"
    }
    $published += $name

    # Deliberate spacing: bursts are what trip PyPI's new-project limiter.
    Start-Sleep -Seconds 90
}

if ($published.Count -gt 0) {
    $record = @()
    if (Test-Path $statePath) {
        $record = @(Get-Content $statePath -Raw -Encoding utf8 | ConvertFrom-Json)
    }
    foreach ($n in $published) {
        $record += [pscustomobject]@{ name = $n; published = (Get-Date -Format "yyyy-MM-dd HH:mm:ss") }
    }
    $record | ConvertTo-Json -Depth 4 | Set-Content -Path $statePath -Encoding utf8
}

Write-Log ("published: {0}" -f $(if ($published.Count) { $published -join ", " } else { "none" }))
Write-Log ("failed:    {0}" -f $(if ($failed.Count)    { $failed -join ", " }    else { "none" }))
Write-Log "=== run end ==="
