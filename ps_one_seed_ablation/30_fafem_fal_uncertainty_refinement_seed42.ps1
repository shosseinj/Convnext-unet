param(
    [ValidateSet(24, 32)][int] $BatchSize = 32,
    [switch] $DryRun
)

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -match '[\\/]main_torch\.py(?:\s|$)'
        } | Select-Object -First 1
    if ($activeTraining) {
        throw "Another GPU training process is active (PID $($activeTraining.ProcessId))."
    }
}

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_30_fafem_fal_uncertainty_refinement" `
    -OutputName "30_fafem_fal_uncertainty_refinement" `
    -SkipMode normal -DeepSupervisionHeads 0 `
    -DeepSupervisionSchedule constant -SamplingMode uniform `
    -EnableFrequencyAugmentation $true `
    -UncertaintyRefinementVersion v2 `
    -BatchSize $BatchSize -Seed 42 -Epochs 180 `
    -DecoderWarmupEpochs 5 -UnfreezeSchedule none `
    -LrScheduler warmup_cosine -LrWarmupEpochs 5 `
    -LearningRate 3e-4 -MinimumLearningRate 1e-6 `
    -EarlyStopPatience 35 -EarlyStopStartEpoch 144 `
    -OptimizerProfile layerwise_convnext -EncoderLayerDecay 0.8 `
    -WeightDecay 1e-2 -EncoderWeightDecay 5e-2 `
    -NewLayerWeightDecay 1e-2 -MaxGradNorm 1.0 `
    -EnableMSC $false -EnableUGBR $false -UpsampleMode bilinear `
    -DetailChannels 0 -DetailFusionMode none -EnableCSAF $false `
    -EnableFAFEM $true -FAFEMStage1 $false -FAFEMStage2 $false `
    -FAFEMStage3 $false -EnableCrossLevelFusion $false `
    -CrossLevelFusionVersion v1 -DryRun:$DryRun
