param([int[]] $Seeds = @(42, 6543, 7777), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "02_add_msc" -EnableMsc $true -SkipMode normal -DetailChannels 0 -EnableGdf $false -DetailFusionMode none -DeepSupervisionHeads 0 -Seeds $Seeds -DryRun:$DryRun
