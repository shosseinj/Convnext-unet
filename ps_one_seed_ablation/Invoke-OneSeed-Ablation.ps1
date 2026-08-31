param(
    [Parameter(Mandatory = $true)][string] $Experiment,
    [Parameter(Mandatory = $true)][string] $OutputName,
    [Parameter(Mandatory = $true)][ValidateSet("normal", "attention_gate", "bsei")][string] $SkipMode,
    [Parameter(Mandatory = $true)][ValidateSet(0, 2)][int] $DeepSupervisionHeads,
    [ValidateSet("constant", "anneal")][string] $DeepSupervisionSchedule = "constant",
    [int] $DeepSupervisionAnnealStart = 96,
    [int] $DeepSupervisionAnnealEnd = 128,
    [ValidateSet("uniform", "lesion_size_weighted")][string] $SamplingMode = "uniform",
    [bool] $EnableFrequencyAugmentation = $false,
    [double] $FrequencyMaxProbability = 0.5,
    [double] $FrequencyMaxMix = 0.5,
    [double] $FrequencyRegionMin = 0.01,
    [double] $FrequencyRegionMax = 0.05,
    [double] $FrequencyConstantFraction = 0.70,
    [double] $FrequencyAnnealEndFraction = 0.80,
    [ValidateSet("none", "v2")][string] $UncertaintyRefinementVersion = "none",
    [int] $Seed = 42,
    [ValidateSet(16, 20, 24, 32, 40, 50)][int] $BatchSize = 24,
    [int] $DecoderWarmupEpochs = 80,
    [int] $UnfreezePlateauPatience = 10,
    [int] $LrPlateauPatience = 5,
    [double] $LrPlateauFactor = 0.9,
    [int] $Epochs = 350,
    [double] $LearningRate = 4e-4,
    [double] $MinimumLearningRate = 1e-6,
    [ValidateSet("plateau", "cosine_warm_restarts", "warmup_cosine")][string] $LrScheduler = "plateau",
    [int] $LrWarmupEpochs = 12,
    [int] $CosineT0 = 8,
    [int] $CosineTMult = 2,
    [int] $EarlyStopPatience = 30,
    [int] $EarlyStopStartEpoch = 0,
    [ValidateSet("plateau", "fixed", "none")][string] $UnfreezeSchedule = "plateau",
    [int[]] $FixedUnfreezeEpochs = @(16, 46, 76, 106, 136),
    [bool] $EnableMSC = $false,
    [bool] $EnableUGBR = $false,
    [ValidateSet("bilinear", "dysample")][string] $UpsampleMode = "bilinear",
    [ValidateSet(0, 32)][int] $DetailChannels = 0,
    [ValidateSet("none", "concatenation")][string] $DetailFusionMode = "none",
    [bool] $EnableCSAF = $false,
    [bool] $EnableFAFEM = $false,
    [bool] $FAFEMStage1 = $false,
    [bool] $FAFEMStage2 = $false,
    [bool] $FAFEMStage3 = $false,
    [bool] $EnableGatedSkipStage3 = $false,
    [bool] $EnableCrossLevelFusion = $false,
    [ValidateSet("v1", "v2")][string] $CrossLevelFusionVersion = "v1",
    [bool] $EnableGeometryConvStage3 = $false,
    [ValidateSet(96, 120)][int] $DecoderHighresWidth = 96,
    [bool] $EnableMSCBLiteStage3 = $false,
    [bool] $EnableFGMSCBLiteStage3 = $false,
    [bool] $EnableResidualFGMSCBLiteStage3 = $false,
    [double] $ResidualFGMSCBGuidanceInitStd = 1e-3,
    [bool] $ResidualFGMSCBSignedStrength = $false,
    [double] $ResidualFGMSCBInitialStrength = 0.0,
    [bool] $EnableDeformableResidualFGMSCBLiteStage3 = $false,
    [bool] $EnableResidualFGMSCBAllSkips = $false,
    [bool] $EnablePartialDeformableResidualFGMSCBLiteStage3 = $false,
    [bool] $EnableF4F3ContextGuidedMSCBLiteStage3 = $false,
    [bool] $EnableMixStyleStage1Stage2 = $false,
    [bool] $EnableMSCBLiteStage2 = $false,
    [bool] $EnableMSCBLiteStage1 = $false,
    [bool] $EnableLKALiteStage3 = $false,
    [ValidateSet("v1", "v2")][string] $CSAFVersion = "v1",
    [ValidateSet("standard", "layerwise_convnext")][string] $OptimizerProfile = "standard",
    [double] $EncoderLayerDecay = 0.8,
    [double] $WeightDecay = 1e-4,
    [double] $EncoderWeightDecay = -1,
    [double] $NewLayerWeightDecay = 1e-3,
    [double] $MaxGradNorm = 0,
    [switch] $ContinueTraining,
    [string] $RefinementCheckpointPath = "",
    [switch] $ResumeLrOverride,
    [switch] $ResetPlateauScheduler,
    [switch] $RefinementForceAllTrainable,
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
    "--max_epochs", [string]$Epochs, "--log_path", (Join-Path $seedDir "KvasirSEG-ConvNeXt_log.txt"))
$initialStateCommand = @($stateCommand)
if ($ContinueTraining) { $initialStateCommand += "--allow_completed_resume" }
$trainCommand = @($python, (Join-Path $repoRoot "main_torch.py"),
    "--experiment_name", $Experiment, "--seed", [string]$Seed, "--seed_dir", $seedDir,
    "--best_checkpoint_path", $bestCheckpoint,
    "--training_history_path", (Join-Path $seedDir "training_history.csv"),
    "--training_summary_path", (Join-Path $seedDir "training_summary.json"),
    "--encoder_weights", $encoderWeights, "--logging_dir", $loggingDir,
    "--enable_msc", ([string]$EnableMSC), "--skip_mode", $SkipMode,
    "--unfreeze_schedule", $UnfreezeSchedule,
    "--fixed_unfreeze_epochs", ($FixedUnfreezeEpochs -join ","),
    "--detail_channels", [string]$DetailChannels,
    "--enable_gdf", "False", "--detail_fusion_mode", $DetailFusionMode,
    "--enable_ugbr", ([string]$EnableUGBR), "--upsample_mode", $UpsampleMode,
    "--enable_csaf", ([string]$EnableCSAF),
    "--enable_fafem", ([string]$EnableFAFEM),
    "--fafem_stage1", ([string]$FAFEMStage1),
    "--fafem_stage2", ([string]$FAFEMStage2),
    "--fafem_stage3", ([string]$FAFEMStage3),
    "--enable_gated_skip_stage3", ([string]$EnableGatedSkipStage3),
    "--enable_cross_level_fusion", ([string]$EnableCrossLevelFusion),
    "--cross_level_fusion_version", $CrossLevelFusionVersion,
    "--enable_geometry_conv_stage3", ([string]$EnableGeometryConvStage3),
    "--decoder_highres_width", [string]$DecoderHighresWidth,
    "--enable_mscb_lite_stage3", ([string]$EnableMSCBLiteStage3),
    "--enable_fg_mscb_lite_stage3", ([string]$EnableFGMSCBLiteStage3),
    "--enable_residual_fg_mscb_lite_stage3", ([string]$EnableResidualFGMSCBLiteStage3),
    "--residual_fg_mscb_guidance_init_std", [string]$ResidualFGMSCBGuidanceInitStd,
    "--residual_fg_mscb_signed_strength", ([string]$ResidualFGMSCBSignedStrength),
    "--residual_fg_mscb_initial_strength", [string]$ResidualFGMSCBInitialStrength,
    "--enable_deformable_residual_fg_mscb_lite_stage3", ([string]$EnableDeformableResidualFGMSCBLiteStage3),
    "--enable_residual_fg_mscb_all_skips", ([string]$EnableResidualFGMSCBAllSkips),
    "--enable_partial_deformable_residual_fg_mscb_lite_stage3", ([string]$EnablePartialDeformableResidualFGMSCBLiteStage3),
    "--enable_f4_f3_context_guided_mscb_lite_stage3", ([string]$EnableF4F3ContextGuidedMSCBLiteStage3),
    "--enable_mixstyle_stage1_stage2", ([string]$EnableMixStyleStage1Stage2),
    "--enable_mscb_lite_stage2", ([string]$EnableMSCBLiteStage2),
    "--enable_mscb_lite_stage1", ([string]$EnableMSCBLiteStage1),
    "--enable_lka_lite_stage3", ([string]$EnableLKALiteStage3),
    "--csaf_version", $CSAFVersion,
    "--deep_supervision_heads", [string]$DeepSupervisionHeads,
    "--deep_supervision_schedule", $DeepSupervisionSchedule,
    "--deep_supervision_anneal_start", [string]$DeepSupervisionAnnealStart,
    "--deep_supervision_anneal_end", [string]$DeepSupervisionAnnealEnd,
    "--sampling_mode", $SamplingMode,
    "--enable_frequency_augmentation", ([string]$EnableFrequencyAugmentation),
    "--frequency_max_probability", [string]$FrequencyMaxProbability,
    "--frequency_max_mix", [string]$FrequencyMaxMix,
    "--frequency_region_min", [string]$FrequencyRegionMin,
    "--frequency_region_max", [string]$FrequencyRegionMax,
    "--frequency_constant_fraction", [string]$FrequencyConstantFraction,
    "--frequency_anneal_end_fraction", [string]$FrequencyAnnealEndFraction,
    "--uncertainty_refinement_version", $UncertaintyRefinementVersion,
    "--epochs", [string]$Epochs, "--batch_size", [string]$BatchSize,
    "--decoder_warmup_epochs", [string]$DecoderWarmupEpochs,
    "--unfreeze_plateau_patience", [string]$UnfreezePlateauPatience,
    "--lr_plateau_patience", [string]$LrPlateauPatience,
    "--lr_plateau_factor", [string]$LrPlateauFactor,
    "--lr_scheduler", $LrScheduler,
    "--warmup_epochs", [string]$LrWarmupEpochs,
    "--cosine_t0", [string]$CosineT0,
    "--cosine_t_mult", [string]$CosineTMult,
    "--optimizer_profile", $OptimizerProfile,
    "--encoder_layer_decay", [string]$EncoderLayerDecay,
    "--encoder_weight_decay", [string]$EncoderWeightDecay,
    "--max_grad_norm", [string]$MaxGradNorm,
    "--amp", "True",
    "--focal_tversky_after_warmup", "False", "--focal_tversky_w", "0",
    "--lr", [string]$LearningRate, "--min_lr", [string]$MinimumLearningRate,
    "--weight_decay", [string]$WeightDecay,
    "--new_layer_weight_decay", [string]$NewLayerWeightDecay,
    "--early_stop_patience", [string]$EarlyStopPatience,
    "--early_stop_start_epoch", [string]$EarlyStopStartEpoch,
    "--training", "True", "--testing", "False", "--tta_check", "False",
    "--load", "False", "--save", "True", "--auto_resume", "True",
    "--resume_optimizer", "True",
    "--resume_lr_override", ([string][bool]$ResumeLrOverride),
    "--reset_plateau_scheduler", ([string][bool]$ResetPlateauScheduler),
    "--refinement_force_all_trainable", ([string][bool]$RefinementForceAllTrainable),
    "--allow_completed_resume", ([string][bool]$ContinueTraining))
if ($RefinementCheckpointPath) {
    $trainCommand += @("--refinement_checkpoint_path", $RefinementCheckpointPath)
}
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
Write-Host "[seed $Seed][$OutputName] FAFEM bottleneck: $(if ($EnableFAFEM) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] FAFEM Stage 3: $(if ($FAFEMStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Gated skip Stage 3: $(if ($EnableGatedSkipStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Decoder high-resolution width: $DecoderHighresWidth"
Write-Host "[seed $Seed][$OutputName] MSCB-lite after Stage-3 fusion: $(if ($EnableMSCBLiteStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Frequency-guided MSCB-lite after Stage-3 fusion: $(if ($EnableFGMSCBLiteStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Residual frequency-guided MSCB-lite after Stage-3 fusion: $(if ($EnableResidualFGMSCBLiteStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Deformable residual FG-MSCB-lite after Stage-3 fusion: $(if ($EnableDeformableResidualFGMSCBLiteStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Residual FG-MSCB-lite on all encoder skips: $(if ($EnableResidualFGMSCBAllSkips) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] MixStyle after encoder Stages 1+2: $(if ($EnableMixStyleStage1Stage2) { 'ON (p=0.5, alpha=0.1; train only)' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] MSCB-lite after Stage-2 fusion: $(if ($EnableMSCBLiteStage2) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] MSCB-lite after Stage-1 fusion: $(if ($EnableMSCBLiteStage1) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] LKA-lite on Stage-3 encoder skip: $(if ($EnableLKALiteStage3) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] FAFEM Stage 2: $(if ($FAFEMStage2) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] FAFEM Stage 1: $(if ($FAFEMStage1) { 'ON' } else { 'OFF' })"
Write-Host "[seed $Seed][$OutputName] Cross-Level Fusion: $(if ($EnableCrossLevelFusion) { 'ON' } else { 'OFF' })"
if ($EnableCrossLevelFusion) {
    Write-Host "[seed $Seed][$OutputName] Cross-Level Fusion version: $CrossLevelFusionVersion"
}
Write-Host "[seed $Seed][$OutputName] MSC: $(if ($EnableMSC) { 'ON' } else { 'OFF' }) | UGBR: $(if ($EnableUGBR) { 'ON' } else { 'OFF' }) | Upsampling: $UpsampleMode | Detail channels: $DetailChannels"
Write-Host "[seed $Seed][$OutputName] Encoder unfreeze schedule: $UnfreezeSchedule$(if ($UnfreezeSchedule -eq 'fixed') { ' (epochs ' + ($FixedUnfreezeEpochs -join ',') + ')' } else { '' })"
Write-Host "[seed $Seed][$OutputName] Output directory: $seedDir"
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
