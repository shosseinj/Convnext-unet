param(
    [ValidateSet(16, 20, 24, 32, 40, 50)][int] $BatchSize = 24,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$experiment = "one_seed_53_fafem_residual_frequency_guided_mscb_stage3_pranet_split_3seeds"
$outputName = "53_fafem_residual_frequency_guided_mscb_stage3_pranet_split_3seeds"
$seeds = @(42, 6543, 7777)
$repoRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repoRoot "configs\splits\development_seed_42.json"
$campaignLog = Join-Path $repoRoot "one_seed_results\ablation\$outputName\CAMPAIGN_CONSOLE.log"

if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Required fixed seen-test manifest is missing: $manifest"
}

Write-Host "Model source: Exp.45 / 45_fafem_residual_frequency_mscb_stage3"
Write-Host "Kvasir: train/validation pool = 900; final test = 100"
Write-Host "ClinicDB: train/validation pool = 550; final test = 62"
Write-Host "Combined train/validation pool: total = 1450; train = 1305; validation = 145"
Write-Host "External tests: CVC-300 = 60; CVC-ColonDB = 380; ETIS = 196"
Write-Host "Split manifest: $manifest"

foreach ($seed in $seeds) {
    $arguments = @{
        Experiment = $experiment
        OutputName = $outputName
        SkipMode = "normal"
        DeepSupervisionHeads = 0
        BatchSize = $BatchSize
        Seed = $seed
        Epochs = 200
        DecoderWarmupEpochs = 0
        UnfreezeSchedule = "none"
        LrScheduler = "warmup_cosine"
        LrWarmupEpochs = 5
        LearningRate = 3e-4
        MinimumLearningRate = 1e-6
        OptimizerProfile = "layerwise_convnext"
        EncoderLayerDecay = 0.8
        WeightDecay = 1e-4
        EncoderWeightDecay = 5e-2
        NewLayerWeightDecay = 1e-2
        MaxGradNorm = 1.0
        EnableCSAF = $false
        EnableFAFEM = $true
        FAFEMStage1 = $false
        FAFEMStage2 = $false
        FAFEMStage3 = $false
        EnableMSC = $false
        EnableUGBR = $false
        EnableGatedSkipStage3 = $false
        EnableCrossLevelFusion = $false
        EnableGeometryConvStage3 = $false
        EnableMSCBLiteStage3 = $false
        EnableFGMSCBLiteStage3 = $false
        EnableResidualFGMSCBLiteStage3 = $true
        ResidualFGMSCBGuidanceInitStd = 0.0
        ResidualFGMSCBSignedStrength = $false
        ResidualFGMSCBInitialStrength = 0.05
        EnableMSCBLiteStage2 = $false
        EnableMSCBLiteStage1 = $false
        EnableLKALiteStage3 = $false
        EnableFrequencyAugmentation = $false
        UncertaintyRefinementVersion = "none"
        SplitProtocol = "pranet_seen_test_internal_validation_v1"
        SplitManifestPath = $manifest
        DryRun = $DryRun
    }
    Write-Host "Launching seed $seed with fixed seen-test protocol."
    if ($DryRun) {
        & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") @arguments
    } else {
        & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") @arguments |
            Tee-Object -FilePath $campaignLog -Append
    }
    if ($LASTEXITCODE -ne 0) { throw "Seed $seed failed with exit code $LASTEXITCODE." }
}
