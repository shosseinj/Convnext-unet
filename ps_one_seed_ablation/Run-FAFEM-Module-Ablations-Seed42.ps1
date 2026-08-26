param([ValidateSet(16, 20, 24)][int] $BatchSize = 24, [switch] $DryRun)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$consoleLog = Join-Path $repoRoot "CAMPAIGN_CONSOLE.log"
$runners = @(
    "07_fafem_plus_ugbr.ps1",
    "08_fafem_plus_dysample.ps1",
    "09_fafem_plus_gated_skips.ps1",
    "10_fafem_plus_detail_branch.ps1",
    "11_fafem_plus_msc.ps1"
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
        Write-Host "[$(Get-Date -Format o)] Running seed 42: $runnerName"
        & $runner -Seed 42 -BatchSize $BatchSize -DryRun:$DryRun
        if ($LASTEXITCODE -ne 0) { throw "Runner failed with exit code $LASTEXITCODE`: $runnerName" }
    }
}
finally {
    if (-not $DryRun) { Stop-Transcript | Out-Null }
}
