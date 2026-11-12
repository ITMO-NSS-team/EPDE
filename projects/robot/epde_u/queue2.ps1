# Nov 12: one lane of the robot revision. Every lane reads the SAME queue file (queue_v2.txt; lines
# <Arm> <Form> <TargetTime 0/1> <Sys> <Epochs> [<cap>]) and claims a run by creating claims\<prefix> (New-Item fails
# when the file exists, so two lanes never take one run). A prefix already in runs.tsv is skipped too. A STOP file
# stops every lane after its current run. All runs import epde from the frozen snapshot (-Repo).
#   queue2.ps1 -Lane <n> [-Queue queue_v2.txt] [-MinFree 4]
# Nov 12 (unit circle): flag column '0' plain, '1' target_axes 0, 'I' invariant_terms 'refuse' on the second snapshot
# (-RepoInv: 9bd360e + the invariant rule only). Lines with flag I belong in a queue only this version reads.
# Portable copy: -Repo / -RepoInv '' = the checkout this folder is in (then it must hold 9bd360e + the invariant
# patch for flag I lines); give two `git archive` snapshot folders to run both arms from one queue.
param([int]$Lane = 1, [string]$Queue = 'queue_v2.txt', [double]$MinFree = 4,
      [string]$Repo = '', [string]$RepoInv = '')
$d = $PSScriptRoot
New-Item -ItemType Directory -Force "$d\claims" | Out-Null
while ($true) {
    if (Test-Path "$d\STOP") { "lane $Lane STOP file: stopped"; break }
    $done = @{}
    if (Test-Path "$d\runs.tsv") {
        foreach ($l in Get-Content "$d\runs.tsv") { $f = $l -split "`t"; if ($f[0] -notlike '*_smoke') { $done[$f[0]] = $true } }
    }
    $next = $null
    foreach ($l in Get-Content "$d\$Queue") {
        $l = $l.Trim(); if (-not $l -or $l.StartsWith('#')) { continue }
        $f = $l -split '\s+'
        $prefix = "$($f[3])_trace_$($f[0])_e$($f[4])"
        if ($done.ContainsKey($prefix) -or (Test-Path "$d\claims\$prefix")) { continue }
        try { New-Item -ItemType File "$d\claims\$prefix" -ErrorAction Stop | Out-Null } catch { continue }
        $next = $f; break
    }
    if ($null -eq $next) { "lane $Lane QUEUE DONE $(Get-Date -Format s)"; break }
    $sel = if ($next.Count -ge 6) { [int]$next[5] } else { 0 }
    "$(Get-Date -Format s) lane $Lane START $($next -join ' ')"
    $inv = $next[2] -eq 'I'
    & "$d\run_u.ps1" -Sys $next[3] -Tag "_$($next[0])_e$($next[4])" -Epochs ([int]$next[4]) -Selected $sel `
        -MinFree $MinFree -Form $next[1] -Repo $(if ($inv) { $RepoInv } else { $Repo }) `
        -TargetTime:($next[2] -eq '1') -Invariant:$inv
    $prefix = "$($next[3])_trace_$($next[0])_e$($next[4])"
    $last = @(Get-Content "$d\runs.tsv" | Where-Object { $_ -like "$prefix`t*" })[-1]
    if ($last -notmatch "exit 0") { "FAILED: $last -- continuing with the next line"; Add-Content "$d\failed.txt" $last }
}
