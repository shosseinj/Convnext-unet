param([int[]] $Seeds = @(42, 6543, 7777), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "09_gdf_concat" -EnableMsc $true -SkipMode bsei -DetailChannels 32 -EnableGdf $false -DetailFusionMode concatenation -DeepSupervisionHeads 3 -Seeds $Seeds -DryRun:$DryRun
