param([ValidateSet(16, 20, 24)][int] $BatchSize = 24, [int] $Seed = 42, [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment "one_seed_01_baseline" -OutputName "01_baseline" `
    -SkipMode normal -DeepSupervisionHeads 0 -BatchSize $BatchSize -Seed $Seed `
    -DecoderWarmupEpochs 15 -UnfreezePlateauPatience 8 `
    -LrPlateauPatience 12 -DryRun:$DryRun
