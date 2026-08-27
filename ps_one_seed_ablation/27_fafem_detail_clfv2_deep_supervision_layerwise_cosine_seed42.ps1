param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessId -ne $PID -and
            $_.CommandLine -match '[\\/]main_torch\.py(?:\s|$)'
        } |
        Select-Object -First 1
    if ($activeTraining) {
        throw "Another GPU training process is active (PID $($activeTraining.ProcessId)). Finish or stop it before starting experiment 27."
    }
}

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_27_fafem_detail_clfv2_deep_supervision_layerwise_cosine" `
    -OutputName "27_fafem_detail_clfv2_deep_supervision_layerwise_cosine" `
    -SkipMode normal `
    -DeepSupervisionHeads 2 `
    -BatchSize $BatchSize `
    -Seed 42 `
    -Epochs 220 `
    -DecoderWarmupEpochs 0 `
    -UnfreezeSchedule none `
    -LrScheduler warmup_cosine `
    -LrWarmupEpochs 5 `
    -LearningRate 3e-4 `
    -MinimumLearningRate 1e-6 `
    -EarlyStopPatience 0 `
    -OptimizerProfile layerwise_convnext `
    -EncoderLayerDecay 0.8 `
    -WeightDecay 1e-2 `
    -EncoderWeightDecay 5e-2 `
    -NewLayerWeightDecay 1e-2 `
    -MaxGradNorm 1.0 `
    -EnableMSC $false `
    -EnableUGBR $false `
    -UpsampleMode bilinear `
    -DetailChannels 32 `
    -DetailFusionMode concatenation `
    -EnableCSAF $false `
    -EnableFAFEM $true `
    -FAFEMStage1 $false `
    -FAFEMStage2 $false `
    -FAFEMStage3 $false `
    -EnableCrossLevelFusion $true `
    -CrossLevelFusionVersion v2 `
    -DryRun:$DryRun
