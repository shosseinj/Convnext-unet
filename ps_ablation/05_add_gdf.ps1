param([int[]] $Seeds = @(42, 3407, 2026), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "05_add_gdf" -EnableMsc $true -SkipMode bsei -DetailChannels 32 -EnableGdf $true -DetailFusionMode gdf -DeepSupervisionHeads 0 -Seeds $Seeds -DryRun:$DryRun
