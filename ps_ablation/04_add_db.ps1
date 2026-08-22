param([int[]] $Seeds = @(42, 3407, 2026), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "04_add_db" -EnableMsc $true -SkipMode bsei -DetailChannels 32 -EnableGdf $false -DetailFusionMode concatenation -DeepSupervisionHeads 0 -Seeds $Seeds -DryRun:$DryRun
