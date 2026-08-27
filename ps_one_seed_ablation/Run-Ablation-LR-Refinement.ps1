param(
    [Parameter(Mandatory = $true)][string] $Experiment,
    [string] $SourceOutputName = "",
    [string] $RefinementOutputName = "",
    [int] $Seed = 6543,
    [ValidateSet(16, 20, 24)][int] $BatchSize = 24,
    [int] $AdditionalEpochs = 56,
    [double] $LearningRate = 1e-4,
    [double] $MinimumLearningRate = 1e-6,
    [int] $LrPlateauPatience = 8,
    [double] $LrPlateauFactor = 0.5,
    [ValidateSet("plateau", "cosine_warm_restarts")][string] $LrScheduler = "plateau",
    [int] $CosineT0 = 8,
    [int] $CosineTMult = 2,
    [int] $EarlyStopPatience = 30,
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

if (-not $SourceOutputName) {
    $SourceOutputName = $Experiment -replace '^one_seed_', ''
}
if (-not $RefinementOutputName) {
    $RefinementOutputName = $SourceOutputName + "_lr_refinement"
}
if ($RefinementOutputName -eq $SourceOutputName) {
    throw "RefinementOutputName must differ from SourceOutputName."
}

$sourceCheckpoint = Join-Path $repoRoot "one_seed_results\ablation\$SourceOutputName\seed_$Seed\best_checkpoint.pth"
if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
    throw "Source checkpoint not found: $sourceCheckpoint"
}

$configJson = & $python (Join-Path $repoRoot "print_ablation_config.py") `
    --experiment $Experiment --checkpoint $sourceCheckpoint --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Could not validate refinement source: $Experiment" }
$config = $configJson | ConvertFrom-Json
if (-not $config.has_optimizer_state -or -not $config.has_ema_state -or -not $config.has_scaler_state) {
    throw "Refinement source must contain optimizer, EMA, and AMP scaler states."
}
$targetEpoch = [int]$config.source_epoch + $AdditionalEpochs

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'main_torch\.py'
    }
    if ($activeTraining) {
        throw "Another GPU training process is active (PID: $($activeTraining.ProcessId -join ', '))."
    }
}

Write-Host "Refinement experiment: $Experiment"
Write-Host "Source checkpoint: $sourceCheckpoint"
Write-Host "Output: one_seed_results\ablation\$RefinementOutputName\seed_$Seed"
Write-Host "Epochs: source=$($config.source_epoch), additional=$AdditionalEpochs, target=$targetEpoch"
Write-Host "LR: encoder=$($LearningRate * 0.1), decoder=$LearningRate, final_refine=$($LearningRate * 1.5)"
Write-Host "Scheduler: $LrScheduler | cosine T_0=$CosineT0 T_mult=$CosineTMult | min_lr=$MinimumLearningRate"

& (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
    -Experiment $Experiment `
    -OutputName $RefinementOutputName `
    -SkipMode $config.skip_mode `
    -DeepSupervisionHeads $config.deep_supervision_heads `
    -BatchSize $BatchSize `
    -Seed $Seed `
    -DecoderWarmupEpochs 15 `
    -UnfreezePlateauPatience 8 `
    -LrPlateauPatience $LrPlateauPatience `
    -LrPlateauFactor $LrPlateauFactor `
    -Epochs $targetEpoch `
    -LearningRate $LearningRate `
    -MinimumLearningRate $MinimumLearningRate `
    -LrScheduler $LrScheduler `
    -CosineT0 $CosineT0 `
    -CosineTMult $CosineTMult `
    -EarlyStopPatience $EarlyStopPatience `
    -UnfreezeSchedule $config.unfreeze_schedule `
    -EnableMSC $config.enable_msc `
    -EnableUGBR $config.enable_ugbr `
    -UpsampleMode $config.upsample_mode `
    -DetailChannels $config.detail_channels `
    -DetailFusionMode $config.detail_fusion_mode `
    -EnableCSAF $config.enable_csaf `
    -EnableFAFEM $config.enable_fafem `
    -FAFEMStage1 $config.fafem_stage1 `
    -FAFEMStage2 $config.fafem_stage2 `
    -FAFEMStage3 $config.fafem_stage3 `
    -EnableCrossLevelFusion $config.enable_cross_level_fusion `
    -CrossLevelFusionVersion $config.cross_level_fusion_version `
    -CSAFVersion $config.csaf_version `
    -RefinementCheckpointPath $sourceCheckpoint `
    -ResumeLrOverride `
    -ResetPlateauScheduler `
    -RefinementForceAllTrainable `
    -DryRun:$DryRun
