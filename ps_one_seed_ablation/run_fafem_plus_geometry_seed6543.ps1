param(
    [ValidateSet(24)][int] $BatchSize = 24,
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
}

Write-Host "seed = 6543"
Write-Host "ImageNet pretrained encoder loaded"
Write-Host "FAFEM bottleneck = ON"
Write-Host "Geometry Conv Stage3 = ON"
Write-Host "encoder frozen = False"
Write-Host "layer decay = 0.75"
Write-Host "warmup = 5"
Write-Host "scheduler = cosine"
Write-Host "grad clip = 1.0"

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_fafem_plus_geometry_conv_stage3" `
    -OutputName "fafem_plus_geometry_conv" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed 6543 `
    -Epochs 120 `
    -DecoderWarmupEpochs 0 `
    -UnfreezeSchedule none `
    -LrScheduler warmup_cosine `
    -LrWarmupEpochs 5 `
    -LearningRate 3e-4 `
    -MinimumLearningRate 1e-6 `
    -OptimizerProfile layerwise_convnext `
    -EncoderLayerDecay 0.75 `
    -WeightDecay 1e-4 `
    -NewLayerWeightDecay 1e-4 `
    -MaxGradNorm 1.0 `
    -EnableFAFEM $true `
    -EnableGeometryConvStage3 $true `
    -EnableMSC $false `
    -EnableUGBR $false `
    -EnableCSAF $false `
    -EnableCrossLevelFusion $false `
    -DetailChannels 0 `
    -DetailFusionMode none `
    -UpsampleMode bilinear `
    -EnableFrequencyAugmentation $false `
    -UncertaintyRefinementVersion none `
    -DryRun:$DryRun
