param([int[]] $Seeds = @(42, 6543, 7777), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "05_add_gdf" -EnableMsc $false -SkipMode normal -DetailChannels 32 -EnableGdf $true -DetailFusionMode gdf -DeepSupervisionHeads 0 -Seeds $Seeds -DryRun:$DryRun
