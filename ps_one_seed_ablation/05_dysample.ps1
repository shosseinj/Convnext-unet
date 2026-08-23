param([switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") -Experiment "one_seed_05_dysample" -OutputName "05_dysample" -SkipMode normal -DeepSupervisionHeads 0 -DryRun:$DryRun
