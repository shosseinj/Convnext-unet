param(
    [Parameter(Mandatory = $true)][string] $Experiment,
    [Parameter(Mandatory = $true)][bool] $EnableMsc,
    [Parameter(Mandatory = $true)][ValidateSet("normal", "bsei")][string] $SkipMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 32)][int] $DetailChannels,
    [Parameter(Mandatory = $true)][bool] $EnableGdf,
    [Parameter(Mandatory = $true)][ValidateSet("none", "addition", "concatenation", "gdf")][string] $DetailFusionMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 3)][int] $DeepSupervisionHeads,
    [int[]] $Seeds = @(42, 3407, 2026),
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$trainer = Join-Path $repoRoot "main_torch.py"
$encoderWeights = Join-Path $repoRoot "convnext_tiny_22k_1k_384.pth"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }
if (-not (Test-Path -LiteralPath $trainer -PathType Leaf)) { throw "Trainer not found: $trainer" }
if (-not (Test-Path -LiteralPath $encoderWeights -PathType Leaf)) { throw "Encoder weights not found: $encoderWeights" }

foreach ($seed in $Seeds) {
    $outputDir = Join-Path $repoRoot "results\ablation\$Experiment\seed_$seed"
    $loggingDir = $outputDir + [IO.Path]::DirectorySeparatorChar
    $command = @(
        $python, $trainer,
        "--seed", [string]$seed,
        "--encoder_weights", $encoderWeights,
        "--logging_dir", $loggingDir,
        "--enable_msc", ([string]$EnableMsc),
        "--skip_mode", $SkipMode,
        "--detail_channels", [string]$DetailChannels,
        "--enable_gdf", ([string]$EnableGdf),
        "--detail_fusion_mode", $DetailFusionMode,
        "--deep_supervision_heads", [string]$DeepSupervisionHeads,
        "--epochs", "150", "--batch_size", "8", "--decoder_warmup_epochs", "10",
        "--lr", "1e-4", "--weight_decay", "1e-4", "--early_stop_patience", "30",
        "--training", "True", "--testing", "False", "--tta_check", "False",
        "--load", "False", "--save", "True"
    )
    if ($DryRun) { $command | ConvertTo-Json -Compress; continue }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    Write-Host "`n========================================"
    Write-Host "Experiment : $Experiment"
    Write-Host "Seed       : $seed"
    Write-Host "Output     : $outputDir"
    Write-Host "========================================"
    $executable = $command[0]
    $arguments = $command[1..($command.Count - 1)]
    & $executable @arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment $Experiment, seed $seed failed with exit code $LASTEXITCODE" }
}
