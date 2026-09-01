param(
    [ValidateSet(24)][int] $BatchSize = 24,
    [ValidateSet(42, 6543, 7777)][int[]] $Seeds = @(42, 6543, 7777),
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$experiment = "one_seed_33_fafem_warmup_cosine"
$outputName = "33_fafem_warmup_cosine"
$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $repoRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }
. (Join-Path $PSScriptRoot "Training-RunnerLease.ps1")

function Get-ExperimentSeedState {
    param([Parameter(Mandatory = $true)][int] $Seed)

    $seedDir = Join-Path $repoRoot "one_seed_results\ablation\$outputName\seed_$Seed"
    $stateArgs = @(
        (Join-Path $repoRoot "ablation_state.py"),
        "--experiment_name", $experiment,
        "--seed", [string]$Seed,
        "--seed_dir", $seedDir,
        "--max_epochs", "200",
        "--log_path", (Join-Path $seedDir "KvasirSEG-ConvNeXt_log.txt")
    )
    $stateJson = & $python @stateArgs
    if ($LASTEXITCODE -ne 0) { throw "State inspection failed for seed $Seed." }
    return ($stateJson | ConvertFrom-Json)
}

function Write-ThreeSeedSummary {
    $states = @(42, 6543, 7777 | ForEach-Object { Get-ExperimentSeedState -Seed $_ })
    if (@($states | Where-Object { $_.training_action -ne "skip" -or -not $_.evaluation_valid }).Count -ne 0) {
        Write-Host "[$outputName] Mean +/- sample std deferred until seeds 42, 6543, and 7777 all have valid evaluations."
        return
    }

    Write-Host "[$outputName] Three-seed mDice mean +/- sample std"
    foreach ($dataset in @("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")) {
        $values = @(42, 6543, 7777 | ForEach-Object {
            $summaryPath = Join-Path $repoRoot "one_seed_results\ablation\$outputName\seed_$_\evaluation_summary.json"
            $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            [double]$summary.results.PSObject.Properties[$dataset].Value.mDice
        })
        $mean = ($values | Measure-Object -Average).Average
        $variance = (($values | ForEach-Object { [math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum) / ($values.Count - 1)
        Write-Host ("  {0}: {1:P3} +/- {2:P3}" -f $dataset, $mean, [math]::Sqrt($variance))
    }
}

if (-not $DryRun) {
    $activeWork = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match '(?i)(main_torch\.py|evaluate\.py)'
    }
    if ($activeWork) {
        throw "Another training or evaluation process is active (PID: $($activeWork.ProcessId -join ', '))."
    }
}

$runnerLease = $null
if (-not $DryRun) {
    $runnerLease = Enter-TrainingRunnerLease -LeasePath (Join-Path $repoRoot ".one_seed_ablation_training.lock")
}

try {
    foreach ($seed in @(42, 6543, 7777)) {
        if ($Seeds -notcontains $seed) { continue }
        $state = Get-ExperimentSeedState -Seed $seed
        if ($state.training_action -eq "skip" -and $state.evaluation_valid) {
            Write-Host "[$outputName][seed $seed] Valid training and evaluation found - skipping."
            continue
        }

        & (Join-Path $PSScriptRoot "Invoke-OneSeed-Ablation.ps1") `
        -Experiment $experiment `
        -OutputName $outputName `
        -SkipMode normal `
        -DeepSupervisionHeads 0 `
        -BatchSize $BatchSize `
        -Seed $seed `
        -Epochs 200 `
        -DecoderWarmupEpochs 0 `
        -UnfreezeSchedule none `
        -LrScheduler warmup_cosine `
        -LrWarmupEpochs 5 `
        -LearningRate 3e-4 `
        -MinimumLearningRate 1e-6 `
        -OptimizerProfile layerwise_convnext `
        -EncoderLayerDecay 0.8 `
        -WeightDecay 1e-4 `
        -EncoderWeightDecay 5e-2 `
        -NewLayerWeightDecay 1e-2 `
        -MaxGradNorm 1.0 `
        -EnableCSAF $false `
        -EnableFAFEM $true `
        -EnableMSC $false `
        -EnableUGBR $false `
        -EnableCrossLevelFusion $false `
        -EnableGeometryConvStage3 $false `
        -EnableFrequencyAugmentation $false `
            -UncertaintyRefinementVersion none `
            -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) { throw "Runner failed for seed $seed with exit code $LASTEXITCODE." }
    }

    Write-ThreeSeedSummary
} finally {
    if ($null -ne $runnerLease) { Exit-TrainingRunnerLease -Lease $runnerLease }
}
