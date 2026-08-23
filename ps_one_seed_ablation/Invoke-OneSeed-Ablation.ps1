param(
    [Parameter(Mandatory = $true)][string] $Experiment,
    [Parameter(Mandatory = $true)][string] $OutputName,
    [Parameter(Mandatory = $true)][ValidateSet("normal", "attention_gate")][string] $SkipMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 2)][int] $DeepSupervisionHeads,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $repoRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }

$seed = 42
$seedDir = Join-Path $repoRoot "one_seed_results\ablation\$OutputName\seed_42"
$loggingDir = $seedDir + [IO.Path]::DirectorySeparatorChar
$bestCheckpoint = Join-Path $seedDir "best_checkpoint.pth"
$evaluationSummary = Join-Path $seedDir "evaluation_summary.json"
$encoderWeights = Join-Path $repoRoot "convnext_tiny_22k_1k_384.pth"
$stateCommand = @($python, (Join-Path $repoRoot "ablation_state.py"),
    "--experiment_name", $Experiment, "--seed", "42", "--seed_dir", $seedDir,
    "--max_epochs", "150", "--log_path", (Join-Path $seedDir "KvasirSEG-ConvNeXt_log.txt"))
$trainCommand = @($python, (Join-Path $repoRoot "main_torch.py"),
    "--experiment_name", $Experiment, "--seed", "42", "--seed_dir", $seedDir,
    "--best_checkpoint_path", $bestCheckpoint,
    "--training_history_path", (Join-Path $seedDir "training_history.csv"),
    "--training_summary_path", (Join-Path $seedDir "training_summary.json"),
    "--encoder_weights", $encoderWeights, "--logging_dir", $loggingDir,
    "--enable_msc", "False", "--skip_mode", $SkipMode, "--detail_channels", "0",
    "--enable_gdf", "False", "--detail_fusion_mode", "none",
    "--deep_supervision_heads", [string]$DeepSupervisionHeads,
    "--epochs", "150", "--batch_size", "24", "--decoder_warmup_epochs", "10",
    "--focal_tversky_after_warmup", "False", "--focal_tversky_w", "0",
    "--lr", "1e-4", "--weight_decay", "1e-4", "--early_stop_patience", "30",
    "--training", "True", "--testing", "False", "--tta_check", "False",
    "--load", "False", "--save", "True", "--auto_resume", "True",
    "--resume_optimizer", "True")
$evaluateCommand = @($python, (Join-Path $repoRoot "evaluate.py"),
    "--experiment_name", $Experiment, "--seed", "42", "--seed_dir", $seedDir,
    "--data_path", (Join-Path $repoRoot "data"), "--encoder_weights", $encoderWeights,
    "--batch_size", "8", "--num_workers", "0", "--output", $evaluationSummary)

if ($DryRun) {
    @{ experiment = $Experiment; output_name = $OutputName; seed_dir = $seedDir;
       train = $trainCommand; evaluate = $evaluateCommand } | ConvertTo-Json -Compress -Depth 4
    exit 0
}

New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
Write-Host "[seed 42][$OutputName] Checking training state..."
$stateJson = & $stateCommand[0] $stateCommand[1..($stateCommand.Count - 1)]
if ($LASTEXITCODE -ne 0) { throw "State inspection failed." }
$state = $stateJson | ConvertFrom-Json
if ($state.training_action -eq "error") { throw $state.training_reason }
if ($state.training_action -in @("train", "resume")) {
    Write-Host "[seed 42][$OutputName] Training incomplete - starting/resuming training."
    & $trainCommand[0] $trainCommand[1..($trainCommand.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
    $stateJson = & $stateCommand[0] $stateCommand[1..($stateCommand.Count - 1)]
    $state = $stateJson | ConvertFrom-Json
    if ($state.training_action -ne "skip") { throw "Training did not produce a completed checkpoint." }
} else {
    Write-Host "[seed 42][$OutputName] Training already complete - skipping training."
}

Write-Host "[seed 42][$OutputName] Checking evaluation state..."
if (-not $state.evaluation_valid) {
    Write-Host "[seed 42][$OutputName] Evaluation missing or invalid - running evaluate.py."
    & $evaluateCommand[0] $evaluateCommand[1..($evaluateCommand.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
    Write-Host "[seed 42][$OutputName] Evaluation complete."
} else {
    Write-Host "[seed 42][$OutputName] Valid evaluation found - skipping evaluation."
}
