param([string]$RepoPattern = "ConvNeXt_Unet")

$ErrorActionPreference = "Stop"
$targets = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and
  $_.CommandLine -like "*$RepoPattern*" -and
  ($_.CommandLine -match "run_experiment_campaign|run_official_queue|train_research|resume_visible")
}

$targets | Sort-Object ProcessId -Descending | ForEach-Object {
  Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2
$remaining = Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -and
  $_.CommandLine -like "*$RepoPattern*" -and
  ($_.CommandLine -match "run_experiment_campaign|run_official_queue|train_research|resume_visible")
}

if ($remaining) {
  $remaining | Select-Object ProcessId, Name, CommandLine | Format-List
  throw "Repository-owned campaign processes remain active."
}

Write-Host "Old campaign processes stopped. Results and checkpoints were not deleted."
