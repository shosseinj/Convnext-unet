param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "../Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_04_fafem_bottleneck_stage3_stage2_stage1" `
    -OutputName "04_fafem_bottleneck_stage3_stage2_stage1" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed 42 `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 `
    -EnableCSAF $false `
    -EnableFAFEM $true `
    -FAFEMStage1 $true `
    -FAFEMStage2 $true `
    -FAFEMStage3 $true `
    -DryRun:$DryRun
