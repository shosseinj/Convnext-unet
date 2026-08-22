param([int[]] $Seeds = @(42, 6543, 7777), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "10_gdf_addition" -EnableMsc $true -SkipMode bsei -DetailChannels 32 -EnableGdf $false -DetailFusionMode addition -DeepSupervisionHeads 3 -Seeds $Seeds -DryRun:$DryRun
