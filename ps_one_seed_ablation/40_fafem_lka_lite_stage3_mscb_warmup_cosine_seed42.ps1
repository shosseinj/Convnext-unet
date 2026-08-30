param(
    [ValidateSet(24)][int] $BatchSize = 24,
    [ValidateSet(42, 6543, 7777)][int[]] $Seeds = @(42, 6543, 7777),
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$experiment = "one_seed_40_fafem_lka_lite_stage3_mscb_warmup_cosine"
$outputName = "40_fafem_lka_lite_stage3_mscb_warmup_cosine"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Training-RunnerLease.ps1")

if (-not $DryRun) {
    $activeWork = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match '(?i)(main_torch\.py|evaluate\.py)'
    }
    if ($activeWork) {
        throw "Another training or evaluation process is active (PID: $($activeWork.ProcessId -join ', '))."
    }
}

$runnerLease = $null
if (-not $DryRun) {
    $runnerLease = Enter-TrainingRunnerLease -LeasePath (Join-Path $repoRoot ".one_seed_ablation_training.lock")
}

try {
    foreach ($seed in @(42, 6543, 7777)) {
        if ($Seeds -notcontains $seed) { continue }
        & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
            -Experiment $experiment `
            -OutputName $outputName `
            -SkipMode normal `
            -DeepSupervisionHeads 0 `
            -BatchSize $BatchSize `
            -Seed $seed `
            -Epochs 200 `
            -DecoderWarmupEpochs 0 `
            -UnfreezeSchedule none `
            -LrScheduler warmup_cosine `
            -LrWarmupEpochs 5 `
            -LearningRate 3e-4 `
            -MinimumLearningRate 1e-6 `
            -OptimizerProfile layerwise_convnext `
            -EncoderLayerDecay 0.8 `
            -WeightDecay 1e-4 `
            -EncoderWeightDecay 5e-2 `
            -NewLayerWeightDecay 1e-2 `
            -MaxGradNorm 1.0 `
            -EnableCSAF $false `
            -EnableFAFEM $true `
            -FAFEMStage1 $false `
            -FAFEMStage2 $false `
            -FAFEMStage3 $false `
            -EnableMSC $false `
            -EnableUGBR $false `
            -EnableGatedSkipStage3 $false `
            -EnableCrossLevelFusion $false `
            -EnableGeometryConvStage3 $false `
            -EnableMSCBLiteStage3 $true `
            -EnableLKALiteStage3 $true `
            -EnableFrequencyAugmentation $false `
            -UncertaintyRefinementVersion none `
            -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) { throw "Runner failed for seed $seed with exit code $LASTEXITCODE." }
    }
} finally {
    if ($null -ne $runnerLease) { Exit-TrainingRunnerLease -Lease $runnerLease }
}
