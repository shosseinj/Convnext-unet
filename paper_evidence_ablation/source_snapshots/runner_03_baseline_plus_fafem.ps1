param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [int] $Seed = 42,
    [switch] $DryRun
)

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_03_baseline_plus_fafem" `
    -OutputName "03_baseline_plus_fafem" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -Seed $Seed `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 `
    -EnableCSAF $false `
    -EnableFAFEM $true `
    -DryRun:$DryRun
