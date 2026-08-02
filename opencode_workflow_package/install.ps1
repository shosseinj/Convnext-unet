[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'
$source = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = [System.IO.Path]::GetFullPath($ProjectRoot)
if (-not (Test-Path $target -PathType Container)) {
    throw "Project directory does not exist: $target"
}

$items = @(
    '.opencode\agents\stop-job.md',
    '.opencode\commands\stop-workflow.md',
    '.opencode\commands\workflow-status.md',
    '.opencode\commands\resume-workflow.md',
    'tools\stop_workflow.ps1'
)

foreach ($relative in $items) {
    $src = Join-Path $source $relative
    $dst = Join-Path $target $relative
    $dstDir = Split-Path -Parent $dst
    New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    Copy-Item -Path $src -Destination $dst -Force
    Write-Output "Installed: $relative"
}

Write-Output ''
Write-Output 'Installation complete.'
Write-Output "Open OpenCode from: $target"
Write-Output 'Available commands: /workflow-status, /stop-workflow, /resume-workflow'
