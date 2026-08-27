param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -match '[\\/]main_torch\.py(?:\s|$)'
        } | Select-Object -First 1
    if ($activeTraining) {
        throw "Another GPU training process is active (PID $($activeTraining.ProcessId))."
    }
    Start-Transcript -Path (Join-Path $PSScriptRoot "..\CAMPAIGN_CONSOLE.log") -Append | Out-Null
}

try {
    & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
        -Experiment "one_seed_31_fafem_mild_fal" `
        -OutputName "31_fafem_mild_fal" `
        -SkipMode normal -DeepSupervisionHeads 0 `
        -BatchSize $BatchSize -Seed 42 -Epochs 350 `
        -DecoderWarmupEpochs 15 -UnfreezePlateauPatience 8 `
        -LrPlateauPatience 12 -LrScheduler plateau `
        -EnableFrequencyAugmentation $true `
        -FrequencyMaxProbability 0.25 -FrequencyMaxMix 0.25 `
        -FrequencyRegionMin 0.01 -FrequencyRegionMax 0.03 `
        -FrequencyConstantFraction 0.60 -FrequencyAnnealEndFraction 0.70 `
        -UncertaintyRefinementVersion none `
        -EnableMSC $false -EnableUGBR $false -UpsampleMode bilinear `
        -DetailChannels 0 -DetailFusionMode none -EnableCSAF $false `
        -EnableFAFEM $true -FAFEMStage1 $false -FAFEMStage2 $false `
        -FAFEMStage3 $false -EnableCrossLevelFusion $false `
        -CrossLevelFusionVersion v1 -DryRun:$DryRun
} finally {
    if (-not $DryRun) {
        Stop-Transcript | Out-Null
    }
}
