param([ValidateSet(16, 20, 24)][int] $BatchSize = 24, [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") -Experiment "one_seed_04_deep_supervision" -OutputName "04_deep_supervision" -SkipMode normal -DeepSupervisionHeads 2 -BatchSize $BatchSize -DryRun:$DryRun
