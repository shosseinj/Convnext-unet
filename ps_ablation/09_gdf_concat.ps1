$ErrorActionPreference = "Stop"

# Replace GDF with direct concatenation.
# Results:
#   results\ablation\09_gdf_concat\seed_42
#   results\ablation\09_gdf_concat\seed_6543
#   results\ablation\09_gdf_concat\seed_7777

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train.py"
$seeds = @(42, 6543, 7777)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = "python"
}

foreach ($seed in $seeds) {
    $outputDir = Join-Path $PSScriptRoot "results\ablation\09_gdf_concat\seed_$seed"

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    Write-Host ""
    Write-Host "========================================"
    Write-Host "Experiment : 09_gdf_concat"
    Write-Host "Seed       : $seed"
    Write-Host "Output     : $outputDir"
    Write-Host "========================================"

    & $python $trainer `
        --seed $seed `
        --output_dir $outputDir `
        --use_msc true `
        --use_lrse true `
        --use_detail_branch true `
        --detail_fusion concat `
        --deep_supervision true

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment 09_gdf_concat, seed $seed failed with exit code $LASTEXITCODE"
    }
}
