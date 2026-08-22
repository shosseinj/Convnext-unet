param(
    [Parameter(Mandatory = $true)][string] $Experiment,
    [Parameter(Mandatory = $true)][bool] $EnableMsc,
    [Parameter(Mandatory = $true)][ValidateSet("normal", "bsei")][string] $SkipMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 32)][int] $DetailChannels,
    [Parameter(Mandatory = $true)][bool] $EnableGdf,
    [Parameter(Mandatory = $true)][ValidateSet("none", "addition", "concatenation", "gdf")][string] $DetailFusionMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 3)][int] $DeepSupervisionHeads,
    [int[]] $Seeds = @(42, 6543, 7777),
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $repoRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }

$trainer = Join-Path $repoRoot "main_torch.py"
$evaluator = Join-Path $repoRoot "evaluate.py"
$summarizer = Join-Path $repoRoot "summarize_seeds.py"
$stateInspector = Join-Path $repoRoot "ablation_state.py"
$encoderWeights = Join-Path $repoRoot "convnext_tiny_22k_1k_384.pth"
foreach ($required in @($trainer, $evaluator, $summarizer, $stateInspector, $encoderWeights)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file not found: $required" }
}

$experimentDir = Join-Path $repoRoot "results\ablation\$Experiment"
$summarizeCommand = @($python, $summarizer, "--experiment_dir", $experimentDir,
    "--experiment_name", $Experiment)

foreach ($seed in $Seeds) {
    $seedDir = Join-Path $experimentDir "seed_$seed"
    $loggingDir = $seedDir + [IO.Path]::DirectorySeparatorChar
    $bestCheckpoint = Join-Path $seedDir "best_checkpoint.pth"
    $history = Join-Path $seedDir "training_history.csv"
    $trainingSummary = Join-Path $seedDir "training_summary.json"
    $logPath = Join-Path $seedDir "KvasirSEG-ConvNeXt_log.txt"
    $evaluationSummary = Join-Path $seedDir "evaluation_summary.json"
    $stateCommand = @($python, $stateInspector, "--experiment_name", $Experiment,
        "--seed", [string]$seed, "--seed_dir", $seedDir, "--max_epochs", "150",
        "--log_path", $logPath)
    $trainCommand = @(
        $python, $trainer,
        "--experiment_name", $Experiment, "--seed", [string]$seed,
        "--seed_dir", $seedDir, "--best_checkpoint_path", $bestCheckpoint,
        "--training_history_path", $history, "--training_summary_path", $trainingSummary,
        "--encoder_weights", $encoderWeights, "--logging_dir", $loggingDir,
        "--enable_msc", ([string]$EnableMsc), "--skip_mode", $SkipMode,
        "--detail_channels", [string]$DetailChannels, "--enable_gdf", ([string]$EnableGdf),
        "--detail_fusion_mode", $DetailFusionMode,
        "--deep_supervision_heads", [string]$DeepSupervisionHeads,
        "--epochs", "150", "--batch_size", "24", "--decoder_warmup_epochs", "10",
        "--lr", "1e-4", "--weight_decay", "1e-4", "--early_stop_patience", "30",
        "--training", "True", "--testing", "False", "--tta_check", "False",
        "--load", "False", "--save", "True", "--auto_resume", "True",
        "--resume_optimizer", "True"
    )
    $evaluateCommand = @($python, $evaluator, "--experiment_name", $Experiment,
        "--seed", [string]$seed, "--seed_dir", $seedDir, "--data_path",
        (Join-Path $repoRoot "data"), "--encoder_weights", $encoderWeights,
        "--batch_size", "8", "--num_workers", "0", "--output", $evaluationSummary)

    if ($DryRun) {
        @{ train = $trainCommand; evaluate = $evaluateCommand; summarize = $summarizeCommand } |
            ConvertTo-Json -Compress -Depth 4
        continue
    }

    New-Item -ItemType Directory -Force -Path $seedDir | Out-Null
    Write-Host "[seed $seed] Checking training state..."
    $stateJson = & $stateCommand[0] $stateCommand[1..($stateCommand.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "[seed $seed] State inspection failed." }
    $state = $stateJson | ConvertFrom-Json
    if ($state.training_action -eq "error") { throw "[seed $seed] $($state.training_reason)" }

    if ($state.training_action -in @("train", "resume")) {
        $message = if ($state.training_action -eq "train") { "No checkpoint found - starting training." } else { "Incomplete checkpoint found - resuming training." }
        Write-Host "[seed $seed] $message"
        & $trainCommand[0] $trainCommand[1..($trainCommand.Count - 1)]
        if ($LASTEXITCODE -ne 0) { throw "[seed $seed] Training failed with exit code $LASTEXITCODE" }
        $stateJson = & $stateCommand[0] $stateCommand[1..($stateCommand.Count - 1)]
        $state = $stateJson | ConvertFrom-Json
        if ($state.training_action -ne "skip") { throw "[seed $seed] Training exited without a valid completed checkpoint: $($state.training_reason)" }
    } else {
        Write-Host "[seed $seed] Training already complete - skipping training."
    }

    Write-Host "[seed $seed] Checking evaluation state..."
    if (-not $state.evaluation_valid) {
        Write-Host "[seed $seed] Evaluation missing or invalid - running evaluate.py."
        & $evaluateCommand[0] $evaluateCommand[1..($evaluateCommand.Count - 1)]
        if ($LASTEXITCODE -ne 0) { throw "[seed $seed] Evaluation failed with exit code $LASTEXITCODE" }
        $stateJson = & $stateCommand[0] $stateCommand[1..($stateCommand.Count - 1)]
        $state = $stateJson | ConvertFrom-Json
        if (-not $state.evaluation_valid) { throw "[seed $seed] Evaluator did not create a valid summary: $($state.evaluation_reason)" }
        Write-Host "[seed $seed] Evaluation complete."
    } else {
        Write-Host "[seed $seed] Valid evaluation found - skipping evaluation."
    }
}

$canonicalSeeds = @(42, 6543, 7777)
if (-not $DryRun -and $Seeds.Count -eq 3 -and -not (Compare-Object ($Seeds | Sort-Object) $canonicalSeeds)) {
    Write-Host "All three seed evaluations are valid - running summarize_seeds.py."
    & $summarizeCommand[0] $summarizeCommand[1..($summarizeCommand.Count - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Aggregation failed with exit code $LASTEXITCODE" }
}
