$ErrorActionPreference = "Stop"

# Baseline + Multi-Scale Context (MSC).
# Results:
#   results\ablation\02_add_msc\seed_42
#   results\ablation\02_add_msc\seed_6543
#   results\ablation\02_add_msc\seed_7777

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train.py"
$seeds = @(42, 6543, 7777)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = "python"
}

foreach ($seed in $seeds) {
    $outputDir = Join-Path $PSScriptRoot "results\ablation\02_add_msc\seed_$seed"

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Experiment : 02_add_msc"
    Write-Host "Seed       : $seed"
    Write-Host "Output     : $outputDir"
    Write-Host "========================================"

    & $python $trainer `
        --seed $seed `
        --output_dir $outputDir `
        --use_msc true `
        --use_lrse false `
        --use_detail_branch false `
        --detail_fusion false `
        --deep_supervision false

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment 02_add_msc, seed $seed failed with exit code $LASTEXITCODE"
    }
}
