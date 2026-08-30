param(
    [ValidateSet(8)][int] $BatchSize = 8,
    [ValidateSet(42, 6543, 7777)][int[]] $Seeds = @(42, 6543, 7777),
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$experiment = "one_seed_39_fafem_mscb_lite_detail_warmup_cosine"
$outputName = "39_fafem_mscb_lite_detail_warmup_cosine"
$repoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $repoRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { $python = "python" }
. (Join-Path $PSScriptRoot "Training-RunnerLease.ps1")

function Test-ValidTtaSummary {
    param([Parameter(Mandatory = $true)][string] $Path, [Parameter(Mandatory = $true)][int] $Seed)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    try {
        $summary = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if ($summary.experiment_name -ne $experiment -or $summary.seed -ne $Seed -or
            $summary.tta -ne $true -or $summary.threshold -ne 0.45) {
            return $false
        }
        foreach ($dataset in @("Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB")) {
            $result = $summary.results.PSObject.Properties[$dataset].Value
            if ($null -eq $result -or $null -eq $result.mDice) { return $false }
        }
        return $true
    } catch { return $false }
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
        $seedDir = Join-Path $repoRoot "one_seed_results\ablation\$outputName\seed_$seed"
        $checkpoint = Join-Path $seedDir "best_checkpoint.pth"
        $output = Join-Path $seedDir "evaluation_summary_tta.json"
        $evaluate = @(
            $python, (Join-Path $repoRoot "evaluate.py"),
            "--experiment_name", $experiment,
            "--seed", [string]$seed,
            "--seed_dir", $seedDir,
            "--data_path", (Join-Path $repoRoot "data"),
            "--encoder_weights", (Join-Path $repoRoot "convnext_tiny_22k_1k_384.pth"),
            "--batch_size", [string]$BatchSize,
            "--num_workers", "0",
            "--tta",
            "--output", $output
        )
        if ($DryRun) {
            @{ experiment = $experiment; seed = $seed; evaluate = $evaluate; output = $output } |
                ConvertTo-Json -Compress -Depth 4
            continue
        }
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Missing completed checkpoint for seed ${seed}: $checkpoint"
        }
        if (Test-ValidTtaSummary -Path $output -Seed $seed) {
            Write-Host "[$outputName][seed $seed] Valid TTA evaluation found - skipping."
            continue
        }
        & $evaluate[0] $evaluate[1..($evaluate.Count - 1)]
        if ($LASTEXITCODE -ne 0) { throw "TTA evaluation failed for seed $seed with exit code $LASTEXITCODE." }
        if (-not (Test-ValidTtaSummary -Path $output -Seed $seed)) {
            throw "TTA evaluation did not produce a valid summary for seed $seed."
        }
    }
} finally {
    if ($null -ne $runnerLease) { Exit-TrainingRunnerLease -Lease $runnerLease }
}
