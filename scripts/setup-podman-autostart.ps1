# Jarvis — Podman machine autostart setup (M50 follow-on).
#
# The Podman machine (WSL2-backed) does NOT start on Windows boot, so M50's
# run_code reports "machine isn't running" after a reboot until someone runs
# `podman machine start` by hand. This drops a windowless launcher into the
# current user's Startup folder so the machine comes up automatically.
#
# Run ONCE (NON-elevated — the Startup folder is user-owned, no admin needed):
#     pwsh -File scripts/setup-podman-autostart.ps1
# Idempotent — safe to re-run (overwrites the launcher in place).
#
# Why the Startup folder, not a scheduled task: the Startup folder is the
# user-scoped "run at logon" mechanism and needs no elevation;
# Register-ScheduledTask is denied to a standard user, and Jarvis runs
# non-elevated. Logon scope is also REQUIRED, not merely convenient — the
# Podman machine is WSL2-backed and WSL2 needs the interactive user session;
# a Session-0 at-startup task would fail (the same lesson as the CUDA STT
# server's onlogon requirement).
#
# Why the .vbs shim: wscript runs windowless and .Run(cmd, 0, False) launches
# podman hidden + async — no console flashes at logon, and logon is not
# blocked for the ~30-60s the machine takes to come up. podman is invoked by
# ABSOLUTE path: a logon launcher must not trust PATH (the M50 stale-PATH
# lesson — a winget PATH update is invisible to an already-running process).

$ErrorActionPreference = 'Stop'

# 1. Resolve podman.exe — PATH first, then the known winget install location.
$podman = (Get-Command podman -ErrorAction SilentlyContinue).Source
if (-not $podman) {
    $known = Join-Path $env:ProgramFiles 'RedHat\Podman\podman.exe'
    if (Test-Path $known) { $podman = $known }
}
if (-not $podman) {
    throw 'podman.exe not found (checked PATH and Program Files). Install Podman first.'
}
Write-Output "podman   : $podman"

# 2. Write the windowless launcher straight into the user's Startup folder —
#    Windows runs everything there at logon, no scheduled task required.
$startup = [Environment]::GetFolderPath('Startup')
$vbsPath = Join-Path $startup 'jarvis-podman-autostart.vbs'
$vbsLine = 'CreateObject("WScript.Shell").Run """' + $podman + '"" machine start", 0, False'
Set-Content -LiteralPath $vbsPath -Value $vbsLine -Encoding ASCII
Write-Output "launcher : $vbsPath"

# 3. Clean up an obsolete launcher location from an earlier setup attempt.
$old = Join-Path (Join-Path $env:LOCALAPPDATA 'Jarvis') 'podman-autostart.vbs'
if (Test-Path $old) {
    Remove-Item -LiteralPath $old -Force
    Write-Output "removed  : $old (obsolete location)"
}

Write-Output ''
Write-Output 'Done. The Podman machine will now start automatically at each logon.'
Write-Output 'Verify after the next reboot with:  podman machine list'
