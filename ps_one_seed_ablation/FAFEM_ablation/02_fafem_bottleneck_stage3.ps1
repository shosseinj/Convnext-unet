param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "../Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_02_fafem_bottleneck_stage3" `
    -OutputName "02_fafem_bottleneck_stage3" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed 42 `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 `
    -EnableCSAF $false `
    -EnableFAFEM $true `
    -FAFEMStage1 $false `
    -FAFEMStage2 $false `
    -FAFEMStage3 $true `
    -DryRun:$DryRun
