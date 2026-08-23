param([switch] $DryRun)
$ErrorActionPreference = "Stop"
foreach ($runner in @("01_baseline.ps1", "02_add_ugbr.ps1", "03_gated_skips.ps1", "04_deep_supervision.ps1", "05_dysample.ps1")) {
    & (Join-Path $PSScriptRoot $runner) -DryRun:$DryRun
}
if (-not $DryRun) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $projectRoot = Split-Path -Parent $repoRoot
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
    if (-not (Test-Path $python)) { $python = "python" }
    & $python (Join-Path $repoRoot "compare_one_seed.py") --root (Join-Path $repoRoot "one_seed_results\ablation")
    if ($LASTEXITCODE -ne 0) { throw "One-seed comparison failed with exit code $LASTEXITCODE" }
}
