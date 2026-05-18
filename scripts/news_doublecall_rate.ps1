<#
.SYNOPSIS
    M47 follow-up instrument: measure how often get_news double-calls into
    web_search, so the "tighten the tool description?" decision is a
    measurement and not a vibe.

.DESCRIPTION
    The M47 news tool covers only five buckets (top/world/tech/business/
    science). For a topic the curated feeds structurally can't cover
    (gaming, sports, a specific company), Claude tends to call get_news
    first, get nothing useful, THEN web_search — a ~5 s wasted round-trip.

    This was a deliberate "don't fix it after one transcript" call (the
    M44.1 discipline: calibrate against measurements, not guesses; and the
    M43 lesson cuts both ways — over-tightening the description risks Claude
    UNDER-using get_news). This script is the measurement that gates the fix.

    The telemetry tell is a single turn whose `[llm] tokens:` line carries
    BOTH `news=yes` AND `web_search=yes`. NOTE the log format gotcha this
    script exists partly to encode: that marker line is appended to the END
    of the `[jarvis] ...` response line — it is NOT at column 0 — so the
    pattern is unanchored `\[llm\] tokens:`, never `^\[llm\]`.

    Honest nuance (why this is a signal to REVIEW, not an auto-trigger):
    both markers on one turn is not always waste. "What's the tech news,
    and is the OpenAI trial still going?" legitimately uses both tools.
    Telemetry alone can't tell "supplemented usefully" from "wasted a
    round-trip" — so the script pairs each double-call with the question
    that triggered it, and you eyeball the sample.

    Decision bar (encoded here so future-you doesn't reconstruct it):
    tighten the get_news description ONLY when EITHER
      * the double-call rate is >= ~20-25% of get_news turns over a week+
        of normal use, OR
      * it is consistently the SAME one or two topics you actually ask
        about regularly (recurrence of a specific miss > a diffuse rate).
    Below that, leave it: answers are still correct, just slower on a
    minority of queries; over-tuning trades a latency papercut for an
    M43-class under-use regression.

.PARAMETER LogPath
    Path to jarvis.log. Defaults to %LOCALAPPDATA%\Jarvis\jarvis.log.

.PARAMETER Sample
    How many double-call examples (question + telemetry tail) to print for
    the eyeball review. Default 8. Use 0 to suppress the sample.

.EXAMPLE
    pwsh ./scripts/news_doublecall_rate.ps1

.EXAMPLE
    pwsh ./scripts/news_doublecall_rate.ps1 -Sample 20

.NOTES
    jarvis.log rotates at 5 MB (src logfile rotation), so this is a RECENT
    window, not all history — which is what you want (recent behavior is
    what the decision is about). Sibling in spirit to scripts/leak_repro.py:
    make the measurement repeatable so the call stays evidence-based.
#>
[CmdletBinding()]
param(
    [string]$LogPath = (Join-Path $env:LOCALAPPDATA 'Jarvis\jarvis.log'),
    [int]$Sample = 8
)

if (-not (Test-Path -LiteralPath $LogPath)) {
    Write-Error "jarvis.log not found at: $LogPath"
    exit 1
}

$lines = Get-Content -LiteralPath $LogPath

$llmTurns    = 0
$newsTurns   = 0
$doubleTurns = 0
# Heuristic pairing: the canonical question line is `[user, <lang>] <text>`
# (both the voice and the text-input paths emit it, so tracking it dedupes
# the `[text-input] received:` echo). We attribute a double-call to the most
# recent question seen — good enough for a review aid, not forensic.
$lastQ = '(question not found in window)'
$hits  = [System.Collections.Generic.List[object]]::new()

foreach ($line in $lines) {
    if ($line -match '^\[user,[^\]]*\]\s*(.+)$') {
        $lastQ = $Matches[1].Trim()
        continue
    }
    if ($line -notmatch '\[llm\] tokens:') { continue }

    $llmTurns++
    $isNews = $line -match 'news=yes'
    $isWeb  = $line -match 'web_search=yes'
    if ($isNews) { $newsTurns++ }
    if ($isNews -and $isWeb) {
        $doubleTurns++
        # Strip the (often huge) `[jarvis] ...` response prefix; keep only
        # the telemetry tail so the sample is readable.
        $tele = ($line -replace '^.*?(\[llm\] tokens:)', '$1').Trim()
        $hits.Add([pscustomobject]@{ Question = $lastQ; Telemetry = $tele })
    }
}

$rate = if ($newsTurns -gt 0) { $doubleTurns / $newsTurns } else { 0 }

Write-Host ''
Write-Host 'M47 get_news -> web_search double-call rate' -ForegroundColor Cyan
Write-Host '-------------------------------------------'
Write-Host "log                : $LogPath"
Write-Host "window             : recent (jarvis.log rotates at 5 MB)"
Write-Host "LLM turns          : $llmTurns"
Write-Host "get_news turns     : $newsTurns"
Write-Host "double-call turns  : $doubleTurns   (news=yes AND web_search=yes)"
if ($newsTurns -eq 0) {
    Write-Host "rate               : n/a  (no get_news turns in this window yet)" -ForegroundColor DarkGray
} else {
    $pct   = '{0:P0}' -f $rate
    $color = if ($rate -ge 0.25) { 'Yellow' } elseif ($rate -ge 0.15) { 'DarkYellow' } else { 'Green' }
    Write-Host "rate               : $pct" -ForegroundColor $color
}

Write-Host ''
Write-Host 'Decision bar: tighten the get_news description ONLY if rate is'
Write-Host '>= ~20-25% over a week+ of normal use, OR the same one/two topics'
Write-Host 'recur. Otherwise leave it (over-tuning risks M43-class under-use).'
Write-Host 'This is a signal to REVIEW the sample below, not an auto-trigger:'
Write-Host 'a turn using both tools can be legit (one question, two needs).'

if ($Sample -gt 0 -and $hits.Count -gt 0) {
    Write-Host ''
    Write-Host "Most recent $([Math]::Min($Sample, $hits.Count)) double-calls (eyeball: was get_news genuinely useless here?):" -ForegroundColor Cyan
    $start = [Math]::Max(0, $hits.Count - $Sample)
    for ($i = $hits.Count - 1; $i -ge $start; $i--) {
        Write-Host ''
        Write-Host ("  Q: " + $hits[$i].Question) -ForegroundColor White
        Write-Host ("  T: " + $hits[$i].Telemetry) -ForegroundColor DarkGray
    }
}
Write-Host ''
