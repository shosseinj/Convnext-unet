param([switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") -Experiment "one_seed_03_gated_skips" -OutputName "03_gated_skips" -SkipMode attention_gate -DeepSupervisionHeads 0 -DryRun:$DryRun
