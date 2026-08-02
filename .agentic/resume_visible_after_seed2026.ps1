$ErrorActionPreference = "Stop"
$repo = "C:\Users\jafari.h\Desktop\ai_project\ConvNeXt_Unet"
$python = "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe"
$validation = Join-Path $repo "results\raw\baseline\seed_2026\official\validation.json"
$pause = Join-Path $repo ".agentic\pause_after_current_run.json"
$status = Join-Path $repo "RUN_STATUS.md"
$console = Join-Path $repo "CAMPAIGN_CONSOLE.log"

while ($true) {
    if (Test-Path -LiteralPath $validation) {
        try {
            $value = Get-Content -LiteralPath $validation -Raw | ConvertFrom-Json
            if ($value.status -eq "PASS" -and $value.variant -eq "baseline" -and $value.seed -eq 2026) { break }
        } catch {}
    }
    Start-Sleep -Seconds 15
}

while (Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "run_official_queue.py|train_research.py"
}) { Start-Sleep -Seconds 5 }

$campaign = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "run_experiment_campaign.py"
}
foreach ($process in ($campaign | Sort-Object ParentProcessId -Descending)) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
while (Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "run_experiment_campaign.py|run_official_queue.py|train_research.py"
}) { Start-Sleep -Seconds 2 }

Remove-Item -LiteralPath $pause -Force
$now = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
@(
    "Stage: EXPERIMENT"
    "Campaign status: PAUSED_FOR_VISIBLE_RESUME"
    "Active run: None"
    "Latest epoch: Complete"
    "Completed runs: 3/18"
    "Last completed run: baseline / seed 2026"
    "Last result: Official validation PASS"
    "Process status: Background campaign stopped; visible resume starting"
    "Last error: None"
    "Next action: Start baseline_msc / seed 42 in the visible terminal"
    "Main console log: $console"
    "Last update: $now"
) | Set-Content -LiteralPath $status -Encoding UTF8

$command = "Set-Location '$repo'; `$env:PYTHONUNBUFFERED='1'; & '$python' -u tools\run_official_queue.py --device cuda --python '$python' 2>&1 | Tee-Object -FilePath '$console' -Append"
Start-Process powershell.exe -ArgumentList @("-NoExit", "-Command", $command)
