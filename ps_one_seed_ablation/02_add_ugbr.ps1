param([ValidateSet(16, 20, 24)][int] $BatchSize = 24, [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") -Experiment "one_seed_02_add_ugbr" -OutputName "02_add_ugbr" -SkipMode normal -DeepSupervisionHeads 0 -BatchSize $BatchSize -DryRun:$DryRun
