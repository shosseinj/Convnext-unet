$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$consoleLog = Join-Path $repoRoot "CAMPAIGN_CONSOLE.log"

function Invoke-LoggedRunner {
    param([Parameter(Mandatory = $true)][string] $Runner)

    & $Runner -Seed 7777 2>&1 | Tee-Object -FilePath $consoleLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Runner failed with exit code $LASTEXITCODE`: $Runner"
    }
}

"[$(Get-Date -Format o)] Starting seed 7777 baseline -> FAFEM campaign." |
    Tee-Object -FilePath $consoleLog -Append
Invoke-LoggedRunner -Runner (Join-Path $PSScriptRoot "01_baseline.ps1")
Invoke-LoggedRunner -Runner (Join-Path $PSScriptRoot "03_baseline_plus_fafem.ps1")
"[$(Get-Date -Format o)] Seed 7777 baseline -> FAFEM campaign complete." |
    Tee-Object -FilePath $consoleLog -Append
