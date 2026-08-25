param(
    [Parameter(Mandatory = $true)][string] $Experiment,
    [Parameter(Mandatory = $true)][string] $OutputName,
    [Parameter(Mandatory = $true)][ValidateSet("normal", "attention_gate")][string] $SkipMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 2)][int] $DeepSupervisionHeads,
    [int] $Seed = 42,
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [int] $DecoderWarmupEpochs = 80,
    [int] $UnfreezePlateauPatience = 10,
    [int] $LrPlateauPatience = 5,
    [bool] $EnableCSAF = $false,
    [bool] $EnableFAFEM = $false,
    [ValidateSet("v1", "v2")][string] $CSAFVersion = "v1",
    [switch] $ContinueTraining,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $repoRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }

$seedDir = Join-Path $repoRoot "one_seed_results\ablation\$OutputName\seed_$Seed"
$loggingDir = $seedDir + [IO.Path]::DirectorySeparatorChar
$bestCheckpoint = Join-Path $seedDir "best_checkpoint.pth"
$evaluationSummary = Join-Path $seedDir "evaluation_summary.json"
$encoderWeights = Join-Path $repoRoot "convnext_tiny_22k_1k_384.pth"
$stateCommand = @($python, (Join-Path $repoRoot "ablation_state.py"),
    "--experiment_name", $Experiment, "--seed", [string]$Seed, "--seed_dir", $seedDir,
    "--max_epochs", "350", "--log_path", (Join-Path $seedDir "KvasirSEG-ConvNeXt_log.txt"))
$initialStateCommand = @($stateCommand)
if ($ContinueTraining) { $initialStateCommand += "--allow_completed_resume" }
$trainCommand = @($python, (Join-Path $repoRoot "main_torch.py"),
    "--experiment_name", $Experiment, "--seed", [string]$Seed, "--seed_dir", $seedDir,
    "--best_checkpoint_path", $bestCheckpoint,
    "--training_history_path", (Join-Path $seedDir "training_history.csv"),
    "--training_summary_path", (Join-Path $seedDir "training_summary.json"),
    "--encoder_weights", $encoderWeights, "--logging_dir", $loggingDir,
    "--enable_msc", "False", "--skip_mode", $SkipMode, "--detail_channels", "0",
    "--enable_gdf", "False", "--detail_fusion_mode", "none",
    "--enable_csaf", ([string]$EnableCSAF),
    "--enable_fafem", ([string]$EnableFAFEM),
    "--csaf_version", $CSAFVersion,
    "--deep_supervision_heads", [string]$DeepSupervisionHeads,
    "--epochs", "350", "--batch_size", [string]$BatchSize,
    "--decoder_warmup_epochs", [string]$DecoderWarmupEpochs,
    "--unfreeze_plateau_patience", [string]$UnfreezePlateauPatience,
    "--lr_plateau_patience", [string]$LrPlateauPatience,
    "--amp", "True",
    "--focal_tversky_after_warmup", "False", "--focal_tversky_w", "0",
    "--lr", "4e-4", "--weight_decay", "1e-4", "--early_stop_patience", "30",
    "--training", "True", "--testing", "False", "--tta_check", "False",
    "--load", "False", "--save", "True", "--auto_resume", "True",
    "--resume_optimizer", "True",
    "--allow_completed_resume", ([string][bool]$ContinueTraining))
$evaluateCommand = @($python, (Join-Path $repoRoot "evaluate.py"),
    "--experiment_name", $Experiment, "--seed", [string]$Seed, "--seed_dir", $seedDir,
    "--data_path", (Join-Path $repoRoot "data"), "--encoder_weights", $encoderWeights,
    "--batch_size", "8", "--num_workers", "0", "--output", $evaluationSummary)

if ($DryRun) {
    @{ experiment = $Experiment; output_name = $OutputName; seed_dir = $seedDir;
       train = $trainCommand; evaluate = $evaluateCommand } | ConvertTo-Json -Compress -Depth 4
    exit 0
}

New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
Write-Host "[seed $Seed][$OutputName] Experiment: $OutputName | Seed: $Seed | CSAF: $($EnableCSAF.ToString().ToLowerInvariant()) | FAFEM: $($EnableFAFEM.ToString().ToLowerInvariant())"
if ($EnableCSAF) {
    Write-Host "[seed $Seed][$OutputName] CSAF version: $CSAFVersion"
}
if ($EnableFAFEM) {
    Write-Host "[seed $Seed][$OutputName] FAFEM placement: bottleneck / encoder stage 4 output"
}
if ($ContinueTraining) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupDir = Join-Path $seedDir "continuation_backups\$timestamp"
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    @($bestCheckpoint, $evaluationSummary,
      (Join-Path $seedDir "training_summary.json"),
      (Join-Path $seedDir "training_history.csv")) | ForEach-Object {
        if (Test-Path -LiteralPath $_ -PathType Leaf) {
            Copy-Item -LiteralPath $_ -Destination $backupDir
        }
    }
    Write-Host "[seed $Seed][$OutputName] Preserved completed artifacts in $backupDir"
}
Write-Host "[seed $Seed][$OutputName] Checking training state..."
$stateJson = & $initialStateCommand[0] $initialStateCommand[1..($initialStateCommand.Count - 1)]
if ($LASTEXITCODE -ne 0) { throw "State inspection failed." }
$state = $stateJson | ConvertFrom-Json
if ($state.training_action -eq "error") { throw $state.training_reason }
if ($state.training_action -in @("train", "resume")) {
    Write-Host "[seed $Seed][$OutputName] Training incomplete - starting/resuming training."
    $ErrorActionPreference = "Continue"
    & $trainCommand[0] $trainCommand[1..($trainCommand.Count - 1)]
    $trainExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($trainExitCode -ne 0) { throw "Training failed with exit code $trainExitCode" }
    $stateJson = & $stateCommand[0] $stateCommand[1..($stateCommand.Count - 1)]
    $state = $stateJson | ConvertFrom-Json
    if ($state.training_action -ne "skip") { throw "Training did not produce a completed checkpoint." }
} else {
    Write-Host "[seed $Seed][$OutputName] Training already complete - skipping training."
}

Write-Host "[seed $Seed][$OutputName] Checking evaluation state..."
if (-not $state.evaluation_valid) {
    Write-Host "[seed $Seed][$OutputName] Evaluation missing or invalid - running evaluate.py."
    $ErrorActionPreference = "Continue"
    & $evaluateCommand[0] $evaluateCommand[1..($evaluateCommand.Count - 1)]
    $evaluateExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($evaluateExitCode -ne 0) { throw "Evaluation failed with exit code $evaluateExitCode" }
    Write-Host "[seed $Seed][$OutputName] Evaluation complete."
} else {
    Write-Host "[seed $Seed][$OutputName] Valid evaluation found - skipping evaluation."
}
