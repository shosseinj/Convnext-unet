param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$campaignLog = Join-Path $repoRoot "CAMPAIGN_CONSOLE.log"

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match '(?i)(^|[\\/\s])main_torch\.py(?:\s|$)'
    }
    if ($activeTraining) {
        throw "Another GPU training process is active (PID: $($activeTraining.ProcessId -join ', '))."
    }
}

$experiments = @(
    "one_seed_19_fafem_clfv2_fixed_unfreeze",
    "one_seed_18_fafem_dysample_detail_fixed_unfreeze",
    "one_seed_17_fafem_msc_fixed_unfreeze",
    "one_seed_16_fafem_detail_branch_fixed_unfreeze",
    "one_seed_14_fafem_dysample_fixed_unfreeze"
)

$transcriptStarted = $false
try {
    if (-not $DryRun) {
        Start-Transcript -LiteralPath $campaignLog -Append | Out-Null
        $transcriptStarted = $true
    }
    foreach ($experiment in $experiments) {
        Write-Host "=== Cosine refinement: $experiment ==="
        & (Join-Path $PSScriptRoot "Run-Ablation-LR-Refinement.ps1") `
            -Experiment $experiment `
            -Seed 6543 `
            -BatchSize $BatchSize `
            -AdditionalEpochs 56 `
            -LearningRate 1e-4 `
            -MinimumLearningRate 1e-6 `
            -LrScheduler cosine_warm_restarts `
            -CosineT0 8 `
            -CosineTMult 2 `
            -EarlyStopPatience 0 `
            -RefinementOutputName (($experiment -replace '^one_seed_', '') + '_cosine_refinement') `
            -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) {
            throw "Cosine refinement failed: $experiment"
        }
    }
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
