param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$seeds = @(42, 7777, 6543)

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -match 'main_torch\.py'
    } | Select-Object -First 1
    if ($activeTraining) {
        Write-Host "Another training process is already running (PID $($activeTraining.ProcessId)). No CLF-v2 campaign was started."
        exit 0
    }
}

foreach ($seed in $seeds) {
    & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
        -Experiment "one_seed_06_fafem_plus_cross_level_fusion_v2" `
        -OutputName "06_fafem_plus_cross_level_fusion_v2" `
        -SkipMode normal `
        -DeepSupervisionHeads 0 `
        -BatchSize $BatchSize `
        -Seed $seed `
        -DecoderWarmupEpochs 15 `
        -UnfreezePlateauPatience 8 `
        -LrPlateauPatience 12 `
        -EnableCSAF $false `
        -EnableFAFEM $true `
        -FAFEMStage1 $false `
        -FAFEMStage2 $false `
        -FAFEMStage3 $false `
        -EnableCrossLevelFusion $true `
        -CrossLevelFusionVersion "v2" `
        -DryRun:$DryRun

    if ($LASTEXITCODE -ne 0) {
        throw "CLF-v2 seed $seed failed with exit code $LASTEXITCODE"
    }
}
