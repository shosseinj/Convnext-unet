param([int[]] $Seeds = @(42, 6543, 7777), [switch] $DryRun)
& (Join-Path $PSScriptRoot "Invoke-Ablation.ps1") -Experiment "04_add_db" -EnableMsc $false -SkipMode normal -DetailChannels 32 -EnableGdf $false -DetailFusionMode concatenation -DeepSupervisionHeads 0 -Seeds $Seeds -DryRun:$DryRun
