$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$consoleLog = Join-Path $repoRoot "CAMPAIGN_CONSOLE.log"

$activeTraining = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(\.exe)?$' -and
    $_.CommandLine -match 'main_torch\.py'
} | Select-Object -First 1
if ($activeTraining) {
    Write-Host "Another training process is already running (PID $($activeTraining.ProcessId)). No seed 6543 campaign was started."
    exit 0
}

function Invoke-LoggedRunner {
    param([Parameter(Mandatory = $true)][string] $Runner)

    & $Runner -Seed 6543
    if ($LASTEXITCODE -ne 0) {
        throw "Runner failed with exit code $LASTEXITCODE`: $Runner"
    }
}

Start-Transcript -Path $consoleLog -Append | Out-Null
try {
    Write-Host "[$(Get-Date -Format o)] Starting seed 6543 baseline -> FAFEM campaign."
    Invoke-LoggedRunner -Runner (Join-Path $PSScriptRoot "01_baseline.ps1")
    Invoke-LoggedRunner -Runner (Join-Path $PSScriptRoot "03_baseline_plus_fafem.ps1")
    Write-Host "[$(Get-Date -Format o)] Seed 6543 baseline -> FAFEM campaign complete."
}
finally {
    Stop-Transcript | Out-Null
}
