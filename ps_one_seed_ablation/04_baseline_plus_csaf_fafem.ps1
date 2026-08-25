param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_04_baseline_plus_csaf_fafem" `
    -OutputName "04_baseline_plus_csaf_fafem" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 `
    -EnableCSAF $true `
    -EnableFAFEM $true `
    -DryRun:$DryRun
