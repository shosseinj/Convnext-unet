param(
    [ValidateSet(24)][int] $BatchSize = 24,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$sourceExperiment = "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine"
$targetExperiment = "one_seed_38_fafem_mscb_lite_stage3_cosine_refinement"
$sourceOutputName = "37_fafem_mscb_lite_stage3_warmup_cosine"
$refinementOutputName = "38_fafem_mscb_lite_stage3_cosine_refinement"
$additionalEpochs = 56
$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $repoRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }
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
        $sourceCheckpoint = Join-Path $repoRoot "one_seed_results\ablation\$sourceOutputName\seed_$seed\best_checkpoint.pth"
        if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
            throw "Source checkpoint not found for seed ${seed}: $sourceCheckpoint"
        }

        $configJson = & $python (Join-Path $repoRoot "print_ablation_config.py") `
            --experiment $sourceExperiment --checkpoint $sourceCheckpoint --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Could not validate source checkpoint for seed $seed." }
        $config = $configJson | ConvertFrom-Json
        if (-not $config.has_optimizer_state -or -not $config.has_ema_state -or -not $config.has_scaler_state) {
            throw "Source checkpoint for seed $seed lacks optimizer, EMA, or AMP scaler state."
        }

        $targetEpoch = [int]$config.source_epoch + $additionalEpochs
        Write-Host "[seed $seed] Experiment 38 refinement from Experiment 37: source epoch=$($config.source_epoch), target epoch=$targetEpoch"

        & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
            -Experiment $targetExperiment `
            -OutputName $refinementOutputName `
            -SkipMode normal `
            -DeepSupervisionHeads 0 `
            -BatchSize $BatchSize `
            -Seed $seed `
            -Epochs $targetEpoch `
            -DecoderWarmupEpochs 0 `
            -UnfreezeSchedule none `
            -LrScheduler cosine_warm_restarts `
            -CosineT0 8 `
            -CosineTMult 2 `
            -LearningRate 1e-4 `
            -MinimumLearningRate 1e-6 `
            -EarlyStopPatience 0 `
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
            -EnableFrequencyAugmentation $false `
            -UncertaintyRefinementVersion none `
            -RefinementCheckpointPath $sourceCheckpoint `
            -ResumeLrOverride `
            -ResetPlateauScheduler `
            -RefinementForceAllTrainable `
            -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) { throw "Refinement runner failed for seed $seed with exit code $LASTEXITCODE." }
    }
} finally {
    if ($null -ne $runnerLease) { Exit-TrainingRunnerLease -Lease $runnerLease }
}
