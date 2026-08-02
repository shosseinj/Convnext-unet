param(
  [string]$Repo = "C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet",
  [string]$Python = "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe",
  [string]$EntryPoint = "tools\run_ugbr_pilot_campaign.py"
)

$ErrorActionPreference = "Stop"
$log = Join-Path $Repo "CAMPAIGN_CONSOLE.log"
$command = @"
Set-Location '$Repo'
`$env:PYTHONUNBUFFERED='1'
& '$Python' -u '$EntryPoint' 2>&1 | Tee-Object -FilePath '$log' -Append
Write-Host ''
Write-Host 'Campaign ended. This window remains open.'
"@

Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $command)
