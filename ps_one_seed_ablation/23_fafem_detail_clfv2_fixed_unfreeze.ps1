param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [int] $Seed = 6543,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_23_fafem_detail_clfv2_fixed_unfreeze" `
    -OutputName "23_fafem_detail_clfv2_fixed_unfreeze" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed $Seed `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 `
    -UnfreezeSchedule fixed `
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
