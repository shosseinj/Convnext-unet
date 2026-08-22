$ErrorActionPreference = "Stop"

# Baseline + MSC + LRS-E.
# Results:
#   results\ablation\03_add_lrse\seed_42
#   results\ablation\03_add_lrse\seed_6543
#   results\ablation\03_add_lrse\seed_7777

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train.py"
$seeds = @(42, 6543, 7777)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = "python"
}

foreach ($seed in $seeds) {
    $outputDir = Join-Path $PSScriptRoot "results\ablation\03_add_lrse\seed_$seed"

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Experiment : 03_add_lrse"
    Write-Host "Seed       : $seed"
    Write-Host "Output     : $outputDir"
    Write-Host "========================================"

    & $python $trainer `
        --seed $seed `
        --output_dir $outputDir `
        --use_msc true `
        --use_lrse true `
        --use_detail_branch false `
        --detail_fusion false `
        --deep_supervision false

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment 03_add_lrse, seed $seed failed with exit code $LASTEXITCODE"
    }
}
