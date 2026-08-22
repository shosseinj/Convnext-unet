param([int[]] $Seeds = @(42, 3407, 2026), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "08_full_without_lrse" -EnableMsc $true -SkipMode normal -DetailChannels 32 -EnableGdf $true -DetailFusionMode gdf -DeepSupervisionHeads 3 -Seeds $Seeds -DryRun:$DryRun
