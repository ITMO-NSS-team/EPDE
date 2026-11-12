# One robot run on EPDE's CURRENT default (Nov 10): plain + NSGA-II resolves to the universal search U (48e3789,
# a1d0ea0). Native: no TRACE_ARM, no TRACE_FACTOR_MOVES (U has its own factor moves); EPDE_RPS_PARAMS carries only
# the variable_order_seed (= the run seed).
#   -Sys <TRACE_SYSTEM, robot_systems.py name>  -Tag <TRACE_TAG>  -Epochs N  [-Selected M] [-Seed s]
# lane env as the Nov 2 campaign: FD accuracy 4 (robot systems override prep), STOP_SHADOW (equal epochs), SELECTION;
# 1 BLAS thread, PYTHONHASHSEED 0, below-normal priority. Waits for >= MinFree GB free RAM (USER Nov 10: "You can run
# sessions as long as they dont crush on memory" -- no process-count gate).
# Appends '<prefix> exit wall cpu peakWS free' to runs.tsv.
# Nov 12: -Repo <dir> imports epde from a frozen snapshot (EPDE_REPO; epde_digest hashes it); -TargetTime adds
# system_rps_params target_axes 0 (plain targets only along the time axis, a0afeaf).
# -Invariant adds system_rps_params invariant_terms 'refuse' (needs a Repo with the rule: 9bd360e +
# invariant_terms_on_9bd360e.patch).
# Portable copy (projects/robot/epde_u): outputs go next to this script; -Repo '' = the checkout this folder is in.
# -Python <python.exe> (or $env:EPDE_PYTHON; default 'python'); -Noise f adds f x std white noise to every field
# (battery systems only, e.g. 0.05).
param([string]$Sys, [string]$Tag, [int]$Epochs, [int]$Selected = 0, [int]$Seed = 0, [double]$MinFree = 6,
      [string]$Form = 'strong', [string]$Repo = '', [switch]$TargetTime, [switch]$Invariant,
      [string]$Python = '', [double]$Noise = 0)
$d = $PSScriptRoot
$python = if ($Python) { $Python } elseif ($env:EPDE_PYTHON) { $env:EPDE_PYTHON } else { 'python' }
Remove-Item Env:EPDE_OPERATORS, Env:TRACE_ARCHIVE_MATING, Env:EPDE_NICHING, Env:EPDE_MATING, Env:TRACE_FORCE_MU, `
    Env:TRACE_SMOKE, Env:EPDE_ROWS, Env:EPDE_STRIDE, Env:PM_TAG, Env:TRACE_SELECTED, Env:PVFP_P, Env:PVFPX_DISK, `
    Env:TRACE_ARM, Env:TRACE_FACTOR_MOVES, Env:PVWT_MODE, Env:PVRVO_MODE, Env:EPDE_FORM, Env:EPDE_NOISE, `
    Env:EPDE_REPO, Env:PVPT_LIBCHECK_IDENTITY, Env:TRACE_PROV, Env:TRACE_RNG -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = '1'; $env:MKL_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'; $env:PYTHONHASHSEED = '0'
$env:TRACE_SELECTION = '1'; $env:TRACE_FD_KWARGS = '{"accuracy": 4}'; $env:TRACE_STOP_SHADOW = '1'
$env:TRACE_SEED = "$Seed"; $env:TRACE_TAG = $Tag; $env:TRACE_EPOCHS = "$Epochs"; $env:TRACE_SYSTEM = $Sys
$rps = @("`"variable_order_seed`": $Seed")
if ($TargetTime) { $rps += "`"target_axes`": 0" }
if ($Invariant) { $rps += "`"invariant_terms`": `"refuse`"" }
$env:EPDE_RPS_PARAMS = '{' + ($rps -join ', ') + '}'
if ($Repo) { $env:EPDE_REPO = $Repo }
if ($Selected -gt 0) { $env:TRACE_SELECTED = "$Selected" }
if ($Form -eq 'weak') { $env:EPDE_FORM = 'weak' }          # Nov 11: weak rows pick their own lattice (no stride)
elseif ($Sys -like 'field*') { $env:EPDE_STRIDE = '[2, 1, 1]' }
if ($Noise -gt 0) { $env:EPDE_NOISE = $Noise.ToString([Globalization.CultureInfo]::InvariantCulture) }
$prefix = "$($Sys)_trace$Tag"
Remove-Item "$d\$prefix.*", "$d\$($prefix)_*", "$d\auto_$prefix.*", "$d\auto_$($prefix)_*" -ErrorAction SilentlyContinue
while ($true) {
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($free -ge $MinFree) { break }          # USER Nov 10: parallel sessions allowed while memory holds
    Start-Sleep -Seconds 30
}
$t0 = Get-Date
$p = Start-Process -FilePath $python -ArgumentList "-B", "$d\sys_trace.py" -WorkingDirectory $d `
    -RedirectStandardOutput "$d\$prefix.out" -RedirectStandardError "$d\$prefix.err" -PassThru -NoNewWindow
$null = $p.Handle
$p.PriorityClass = 'BelowNormal'
$peak = 0
while (-not $p.HasExited) {
    Start-Sleep -Seconds 3
    try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
}
$p.WaitForExit()
$wall = ((Get-Date) - $t0).TotalSeconds
$cpu = $p.TotalProcessorTime.TotalSeconds
$line = [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}`texit {1}`twall {2:F1}`tcpu {3:F1}`tpeakWS_MB {4:F0}`tfree_GB {5:F1}", $prefix, $p.ExitCode, $wall, $cpu, ($peak / 1MB), $free)
Add-Content "$d\runs.tsv" $line
$tb = if ((Test-Path "$d\$prefix.err") -and (Select-String -Path "$d\$prefix.err" -Pattern 'Traceback' -Quiet)) { ' TRACEBACK' } else { '' }
"$line$tb"
