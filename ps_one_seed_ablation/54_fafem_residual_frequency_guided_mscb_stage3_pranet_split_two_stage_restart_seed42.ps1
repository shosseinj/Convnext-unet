param(
    [ValidateSet(24)][int] $BatchSize = 24,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$experiment = "one_seed_54_fafem_residual_frequency_guided_mscb_stage3_pranet_split_two_stage_restart"
$outputName = "54_fafem_residual_frequency_guided_mscb_stage3_pranet_split_two_stage_restart"
$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repoRoot "configs\splits\development_seed_42.json"
. (Join-Path $PSScriptRoot "Training-RunnerLease.ps1")

if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Required fixed seen-test manifest is missing: $manifest"
}
if (-not $DryRun) {
    $activeWork = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match '(?i)(main_torch\.py|evaluate\.py)'
    }
    if ($activeWork) {
        throw "Another training or evaluation process is active (PID: $($activeWork.ProcessId -join ', '))."
    }
}

Write-Host "Model source: Exp.53 architecture / Exp.45 residual FG-MSCB"
Write-Host "Protocol: warmup (1-5), cosine (6-120), controlled restart (121-200)"
Write-Host "Early stopping: OFF; checkpoint selection: internal validation IoU"
Write-Host "Split manifest: $manifest"

$runnerLease = $null
if (-not $DryRun) {
    $runnerLease = Enter-TrainingRunnerLease -LeasePath (Join-Path $repoRoot ".one_seed_ablation_training.lock")
}
try {
    & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
        -Experiment $experiment -OutputName $outputName -SkipMode normal `
        -DeepSupervisionHeads 0 -BatchSize $BatchSize -Seed 42 -Epochs 200 `
        -DecoderWarmupEpochs 0 -UnfreezeSchedule none `
        -LrScheduler two_stage_warmup_cosine_restart -LrWarmupEpochs 5 `
        -RestartEpoch 120 -FirstCycleMinFactor 0.1 -RestartFactor (1.0 / 3.0) `
        -LearningRate 3e-4 -MinimumLearningRate 1e-6 `
        -EarlyStopPatience 0 -EarlyStopStartEpoch 0 `
        -OptimizerProfile layerwise_convnext -EncoderLayerDecay 0.8 `
        -WeightDecay 1e-4 -EncoderWeightDecay 5e-2 -NewLayerWeightDecay 1e-2 `
        -MaxGradNorm 1.0 -EnableCSAF $false -EnableFAFEM $true `
        -FAFEMStage1 $false -FAFEMStage2 $false -FAFEMStage3 $false `
        -EnableMSC $false -EnableUGBR $false -EnableGatedSkipStage3 $false `
        -EnableCrossLevelFusion $false -EnableGeometryConvStage3 $false `
        -EnableMSCBLiteStage3 $false -EnableFGMSCBLiteStage3 $false `
        -EnableResidualFGMSCBLiteStage3 $true -ResidualFGMSCBGuidanceInitStd 0.0 `
        -ResidualFGMSCBSignedStrength $false -ResidualFGMSCBInitialStrength 0.05 `
        -EnableMSCBLiteStage2 $false -EnableMSCBLiteStage1 $false `
        -EnableLKALiteStage3 $false -EnableFrequencyAugmentation $false `
        -UncertaintyRefinementVersion none `
        -SplitProtocol pranet_seen_test_internal_validation_v1 -SplitManifestPath $manifest `
        -DryRun:$DryRun
    if ($LASTEXITCODE -ne 0) { throw "Runner failed with exit code $LASTEXITCODE." }
} finally {
    if ($null -ne $runnerLease) { Exit-TrainingRunnerLease -Lease $runnerLease }
}
