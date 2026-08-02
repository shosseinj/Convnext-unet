param(
  [string]$ProjectRoot = (Get-Location).Path
)
$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backup = Join-Path $ProjectRoot ".agentic_opencode_backup_$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($name in @('AGENTS.md','opencode.json','.opencode','agentic_workflow')) {
  $target = Join-Path $ProjectRoot $name
  if (Test-Path $target) {
    $dest = Join-Path $backup $name
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    Copy-Item $target $dest -Recurse -Force
  }
}
Copy-Item (Join-Path $PackageRoot 'AGENTS.md') (Join-Path $ProjectRoot 'AGENTS.md') -Force
Copy-Item (Join-Path $PackageRoot 'opencode.json') (Join-Path $ProjectRoot 'opencode.json') -Force
Copy-Item (Join-Path $PackageRoot '.opencode') (Join-Path $ProjectRoot '.opencode') -Recurse -Force
Copy-Item (Join-Path $PackageRoot 'agentic_workflow') (Join-Path $ProjectRoot 'agentic_workflow') -Recurse -Force
Write-Host "Installed. Existing workflow/config files backed up to: $backup"
Write-Host "Source code, results, checkpoints, and Codex modifications were not changed."
