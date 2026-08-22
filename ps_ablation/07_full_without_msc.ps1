param([int[]] $Seeds = @(42, 3407, 2026), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "07_full_without_msc" -EnableMsc $false -SkipMode bsei -DetailChannels 32 -EnableGdf $true -DetailFusionMode gdf -DeepSupervisionHeads 3 -Seeds $Seeds -DryRun:$DryRun
