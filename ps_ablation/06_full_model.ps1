param([int[]] $Seeds = @(42, 6543, 7777), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "06_full_model" -EnableMsc $true -SkipMode bsei -DetailChannels 32 -EnableGdf $true -DetailFusionMode gdf -DeepSupervisionHeads 3 -Seeds $Seeds -DryRun:$DryRun
