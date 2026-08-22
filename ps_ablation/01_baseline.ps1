param([int[]] $Seeds = @(42, 3407, 2026), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "01_baseline" -EnableMsc $false -SkipMode normal -DetailChannels 0 -EnableGdf $false -DetailFusionMode none -DeepSupervisionHeads 0 -Seeds $Seeds -DryRun:$DryRun
