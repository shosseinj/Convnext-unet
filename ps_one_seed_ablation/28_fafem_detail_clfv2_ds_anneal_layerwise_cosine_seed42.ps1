param(
    [ValidateSet(16, 20, 24, 32)][int] $BatchSize = 24,
    [switch] $DryRun
)

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match '[\\/]main_torch\.py(?:\s|$)' } |
        Select-Object -First 1
    if ($activeTraining) {
        throw "Another GPU training process is active (PID $($activeTraining.ProcessId))."
    }
}

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_28_fafem_detail_clfv2_ds_anneal_layerwise_cosine" `
    -OutputName "28_fafem_detail_clfv2_ds_anneal_layerwise_cosine" `
    -SkipMode normal -DeepSupervisionHeads 2 `
    -DeepSupervisionSchedule anneal -DeepSupervisionAnnealStart 96 `
    -DeepSupervisionAnnealEnd 128 -SamplingMode uniform `
    -BatchSize $BatchSize -Seed 42 -Epochs 160 `
    -DecoderWarmupEpochs 0 -UnfreezeSchedule none `
    -LrScheduler warmup_cosine -LrWarmupEpochs 5 `
    -LearningRate 3e-4 -MinimumLearningRate 1e-6 `
    -EarlyStopPatience 0 -OptimizerProfile layerwise_convnext `
    -EncoderLayerDecay 0.8 -WeightDecay 1e-2 `
    -EncoderWeightDecay 5e-2 -NewLayerWeightDecay 1e-2 `
    -MaxGradNorm 1.0 -EnableMSC $false -EnableUGBR $false `
    -UpsampleMode bilinear -DetailChannels 32 `
    -DetailFusionMode concatenation -EnableCSAF $false `
    -EnableFAFEM $true -FAFEMStage1 $false -FAFEMStage2 $false `
    -FAFEMStage3 $false -EnableCrossLevelFusion $true `
    -CrossLevelFusionVersion v2 -DryRun:$DryRun
