param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_24_fafem_detail_clfv2_optimized_training" `
    -OutputName "24_fafem_detail_clfv2_optimized_training" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed 42 `
    -Epochs 250 `
    -DecoderWarmupEpochs 5 `
    -UnfreezeSchedule fixed `
    -FixedUnfreezeEpochs @(6, 16, 26, 36, 46) `
    -LrScheduler plateau `
    -LrPlateauPatience 8 `
    -LrPlateauFactor 0.5 `
    -LearningRate 2e-4 `
    -MinimumLearningRate 1e-6 `
    -EarlyStopPatience 30 `
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
