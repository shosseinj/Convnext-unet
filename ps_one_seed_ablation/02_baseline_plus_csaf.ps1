param(
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [switch] $DryRun
)

$invokeRunner = Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1"
& $invokeRunner `
    -Experiment "one_seed_02_baseline_plus_csaf" `
    -OutputName "02_baseline_plus_csaf" `
    -SkipMode normal `
    -DeepSupervisionHeads 0 `
    -BatchSize $BatchSize `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 `
    -EnableCSAF $true `
    -DryRun:$DryRun
