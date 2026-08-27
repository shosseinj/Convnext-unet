param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_25_fafem_detail_clfv2_warmup_cosine" `
    -OutputName "25_fafem_detail_clfv2_warmup_cosine" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed 42 `
    -Epochs 200 `
    -DecoderWarmupEpochs 5 `
    -UnfreezeSchedule fixed `
    -FixedUnfreezeEpochs @(6, 16, 26, 36, 46) `
    -LrScheduler warmup_cosine `
    -LrWarmupEpochs 5 `
    -LearningRate 1e-4 `
    -MinimumLearningRate 1e-6 `
    -EarlyStopPatience 0 `
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
