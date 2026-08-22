$ErrorActionPreference = "Stop"

# Mechanism isolation: full model without deep supervision. Same configuration as 05_add_gdf.
# Results:
#   results\ablation\11_full_without_ds\seed_42
#   results\ablation\11_full_without_ds\seed_6543
#   results\ablation\11_full_without_ds\seed_7777

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train.py"
$seeds = @(42, 6543, 7777)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = "python"
}

foreach ($seed in $seeds) {
    $outputDir = Join-Path $PSScriptRoot "results\ablation\11_full_without_ds\seed_$seed"

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Experiment : 11_full_without_ds"
    Write-Host "Seed       : $seed"
    Write-Host "Output     : $outputDir"
    Write-Host "========================================"

    & $python $trainer `
        --seed $seed `
        --output_dir $outputDir `
        --use_msc true `
        --use_lrse true `
        --use_detail_branch true `
        --detail_fusion true `
        --deep_supervision false

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment 11_full_without_ds, seed $seed failed with exit code $LASTEXITCODE"
    }
}
