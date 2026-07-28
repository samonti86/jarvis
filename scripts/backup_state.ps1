<#
.SYNOPSIS
    Replicate Jarvis's durable runtime state to a private offsite repo.

.DESCRIPTION
    %LOCALAPPDATA%\Jarvis\ holds everything the process learns and remembers,
    and NONE of it is in the project repo. Most of it genuinely is rebuildable
    -- logs rotate, diagnostics regenerate, knowledge.db is only an FTS5 cache
    of the jarvis-knowledge repo. Two things are not:

      sessions\ + summaries.jsonl   verbatim conversation history. This is the
                                    corpus M78's recall searches over. There is
                                    no way to reconstruct it. Gone is gone.
      speakers\                     Resemblyzer voice enrollments. Rebuildable
                                    only by having each person re-enroll aloud.

    So this script copies ~675 KB of the 18.6 MB tree and pushes it. It is
    deliberately small: the point is the irreplaceable part, not a disk image.

    THE INCLUDE LIST IS AN ALLOWLIST, NOT A DENYLIST, and that is load-bearing.
    This directory already contains a TLS private key and a Ring API token, and
    the security\ subtree is evidence snapshots -- photographs of whoever
    walked past the camera. A denylist backs those up the day someone adds a
    file the pattern did not anticipate. An allowlist fails closed: a new file
    is simply not copied until someone opts it in here.

.PARAMETER Push
    Commit and push. Without it the script only refreshes the working tree so
    you can review the diff first -- same contract as claude-config's
    Checkpoint.ps1, which this deliberately mirrors.

.EXAMPLE
    pwsh -File scripts\backup_state.ps1 -Push
#>
[CmdletBinding()]
param(
    [switch]$Push,
    [string]$Message = "",
    [string]$Dest = "$env:USERPROFILE\repos\jarvis-state"
)

$ErrorActionPreference = "Stop"
function Write-Warn { param($m) Write-Host $m -ForegroundColor Yellow }

$src = Join-Path $env:LOCALAPPDATA "Jarvis"
if (-not (Test-Path $src)) { throw "No Jarvis state directory at $src" }
if (-not (Test-Path $Dest)) { throw "No backup repo at $Dest (create it first, PRIVATE)" }

# --- the allowlist --------------------------------------------------------
# Relative to %LOCALAPPDATA%\Jarvis. Directories are copied recursively.
# EXCLUDED ON PURPOSE, and why:
#   security\        evidence snapshots -- photos of people. Privacy.
#   diagnostics\     4.7 MB of regenerable dumps.
#   jarvis*.log*     rotating logs; noise, and the largest thing in the tree.
#   knowledge.db     rebuildable cache; truth lives in the jarvis-knowledge repo.
#   tls\             TLS private key.        <- secret
#   ring_token.json  Ring API token.         <- secret
$include = @(
    "sessions",                       # verbatim transcripts  <- irreplaceable
    "speakers",                       # voice enrollments     <- costly to redo
    "summaries.jsonl",                # session summaries     <- irreplaceable
    "reminders.json",
    "predictions.json",
    "background_agent_ids.json",
    "calendar_announced.json",
    "weather_alerts_announced.json",
    "ui_state.json"
)

$stage = Join-Path $Dest "state"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

$copied = 0
foreach ($rel in $include) {
    $from = Join-Path $src $rel
    if (-not (Test-Path $from)) {
        Write-Host "  --   $rel (absent)" -ForegroundColor DarkGray
        continue
    }
    $to = Join-Path $stage $rel
    if (Test-Path $from -PathType Container) {
        Copy-Item $from $to -Recurse -Force
        $n = (Get-ChildItem $to -Recurse -File).Count
    } else {
        Copy-Item $from $to -Force
        $n = 1
    }
    $copied += $n
    Write-Host ("  ==>  {0,-32} {1,4} file(s)" -f $rel, $n) -ForegroundColor Cyan
}

$kb = [math]::Round((Get-ChildItem $stage -Recurse -File | Measure-Object Length -Sum).Sum / 1KB, 1)
Write-Host "Snapshot complete: $copied file(s), $kb KB." -ForegroundColor Green

if (-not $Push) {
    Write-Host "`nWorking tree refreshed. Review, then re-run with -Push." -ForegroundColor DarkGray
    return
}

Push-Location $Dest
try {
    git add -A | Out-Null

    # SECURITY GATE -- second line of defence behind the allowlist. If a secret
    # ever reaches staging, something upstream is already wrong; refuse anyway.
    $staged = git diff --cached --name-only
    $forbidden = $staged | Where-Object {
        $_ -match '(?i)(^|/)tls/|\.key$|\.pem$|\.crt$|\.pfx$|token|credential|secret|\.env'
    }
    if ($forbidden) {
        Write-Warn "SECURITY GATE TRIPPED -- refusing to push. Forbidden staged paths:"
        $forbidden | ForEach-Object { Write-Warn "    $_" }
        throw "Backup aborted: forbidden files staged."
    }

    if (-not $staged) {
        Write-Host "Nothing changed since last backup -- already current." -ForegroundColor Green
        return
    }

    Write-Host "Security gate: CLEAN ($($staged.Count) file(s) staged)." -ForegroundColor Green
    if (-not $Message) { $Message = "State snapshot $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
    git commit -m $Message | Out-Null
    git push
    Write-Host "Backup pushed. Offsite replica is current." -ForegroundColor Green
}
finally {
    Pop-Location
}
