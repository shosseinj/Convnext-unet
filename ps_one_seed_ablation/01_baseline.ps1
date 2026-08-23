param([switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") -Experiment "one_seed_01_baseline" -OutputName "01_baseline" -SkipMode normal -DeepSupervisionHeads 0 -DryRun:$DryRun
