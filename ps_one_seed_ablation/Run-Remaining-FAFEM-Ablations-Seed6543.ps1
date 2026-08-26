param([ValidateSet(16, 20, 24)][int] $BatchSize = 24, [switch] $DryRun)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$consoleLog = Join-Path $repoRoot "CAMPAIGN_CONSOLE.log"
$runners = @(
    "18_fafem_dysample_detail_fixed_unfreeze.ps1",
    "19_fafem_clfv2_fixed_unfreeze.ps1",
    "20_fafem_clfv1_fixed_unfreeze.ps1",
    "21_fafem_dysample_detail_clfv2_fixed_unfreeze.ps1",
    "22_fafem_dysample_detail_clfv1_fixed_unfreeze.ps1"
)

if (-not $DryRun) {
    $activeTraining = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and $_.CommandLine -match 'main_torch\.py'
    } | Select-Object -First 1
    if ($activeTraining) {
        throw "Another GPU training process is already running (PID $($activeTraining.ProcessId))."
    }
    Start-Transcript -Path $consoleLog -Append | Out-Null
}

try {
    foreach ($runnerName in $runners) {
        $runner = Join-Path $PSScriptRoot $runnerName
        Write-Host "[$(Get-Date -Format o)] Remaining FAFEM seed 6543: $runnerName"
        & $runner -Seed 6543 -BatchSize $BatchSize -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) {
            throw "Runner failed with exit code $LASTEXITCODE`: $runnerName"
        }
    }
}
finally {
    if (-not $DryRun) { Stop-Transcript | Out-Null }
}
