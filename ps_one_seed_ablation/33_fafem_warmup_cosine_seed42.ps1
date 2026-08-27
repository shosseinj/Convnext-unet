param(
    [ValidateSet(24)][int] $BatchSize = 24,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_33_fafem_warmup_cosine" `
    -OutputName "33_fafem_warmup_cosine" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed 42 `
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
    -EnableMSC $false `
    -EnableUGBR $false `
    -EnableCrossLevelFusion $false `
    -EnableGeometryConvStage3 $false `
    -EnableFrequencyAugmentation $false `
    -UncertaintyRefinementVersion none `
    -DryRun:$DryRun
