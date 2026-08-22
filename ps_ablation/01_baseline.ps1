$ErrorActionPreference = "Stop"

# Baseline: ConvNeXt-Tiny encoder + decoder + final prediction head only.
# Results:
#   results\ablation\01_baseline\seed_42
#   results\ablation\01_baseline\seed_6543
#   results\ablation\01_baseline\seed_7777

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train.py"
$seeds = @(42, 6543, 7777)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = "python"
}

foreach ($seed in $seeds) {
    $outputDir = Join-Path $PSScriptRoot "results\ablation\01_baseline\seed_$seed"

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Experiment : 01_baseline"
    Write-Host "Seed       : $seed"
    Write-Host "Output     : $outputDir"
    Write-Host "========================================"

    & $python $trainer `
        --seed $seed `
        --output_dir $outputDir `
        --use_msc false `
        --use_lrse false `
        --use_detail_branch false `
        --detail_fusion false `
        --deep_supervision false

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment 01_baseline, seed $seed failed with exit code $LASTEXITCODE"
    }
}
