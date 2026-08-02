[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\\')
$statusPath = Join-Path $repo 'RUN_STATUS.md'
$agenticDir = Join-Path $repo '.agentic'
$lockPaths = @(
    (Join-Path $agenticDir 'experiment_campaign.lock.json'),
    (Join-Path $agenticDir 'official_queue.lock.json')
)
$patterns = @(
    'run_experiment_campaign.py',
    'run_official_queue.py',
    'train_research.py',
    'resume_visible_after_'
)

function Get-WorkflowProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $cmd = [string]$_.CommandLine
        if ([string]::IsNullOrWhiteSpace($cmd)) { return $false }
        $inRepo = $cmd.IndexOf($repo, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
        if (-not $inRepo) { return $false }
        foreach ($pattern in $patterns) {
            if ($cmd.IndexOf($pattern, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
                return $true
            }
        }
        return $false
    }
}

function Get-DepthMap([object[]]$Processes) {
    $byId = @{}
    foreach ($p in $Processes) { $byId[[int]$p.ProcessId] = $p }
    $depth = @{}
    foreach ($p in $Processes) {
        $d = 0
        $seen = @{}
        $parent = [int]$p.ParentProcessId
        while ($byId.ContainsKey($parent) -and -not $seen.ContainsKey($parent)) {
            $seen[$parent] = $true
            $d++
            $parent = [int]$byId[$parent].ParentProcessId
        }
        $depth[[int]$p.ProcessId] = $d
    }
    return $depth
}

function Write-RunStatus([string]$ProcessStatus, [string]$ErrorText) {
    $existing = @{}
    if (Test-Path $statusPath) {
        foreach ($line in Get-Content $statusPath -ErrorAction SilentlyContinue) {
            if ($line -match '^([^:]+):\s*(.*)$') { $existing[$matches[1].Trim()] = $matches[2].Trim() }
        }
    }
    $stage = if ($existing.ContainsKey('Stage')) { $existing['Stage'] } else { 'UNKNOWN' }
    $completed = if ($existing.ContainsKey('Completed runs')) { $existing['Completed runs'] } elseif ($existing.ContainsKey('Completed pilots')) { $existing['Completed pilots'] } else { 'Unknown' }
    $lastCompleted = if ($existing.ContainsKey('Last completed run')) { $existing['Last completed run'] } elseif ($existing.ContainsKey('Last completed pilot')) { $existing['Last completed pilot'] } else { 'Unknown' }
    $lastResult = if ($existing.ContainsKey('Last result')) { $existing['Last result'] } else { 'Preserved' }
    $lines = @(
        "Stage: $stage",
        'Campaign status: STOPPED',
        'Active run: None',
        'Latest epoch: Not running',
        "Completed runs: $completed",
        "Last completed run: $lastCompleted",
        "Last result: $lastResult",
        "Process status: $ProcessStatus",
        "Last error: $ErrorText",
        'Next action: Run /resume-workflow when ready',
        'Console log: CAMPAIGN_CONSOLE.log',
        "Last update: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))"
    )
    Set-Content -Path $statusPath -Value $lines -Encoding UTF8
}

try {
    $processes = @(Get-WorkflowProcesses)
    if ($processes.Count -eq 0) {
        foreach ($lock in $lockPaths) {
            if (Test-Path $lock) { Remove-Item $lock -Force }
        }
        Write-RunStatus -ProcessStatus 'No workflow process was active' -ErrorText 'None'
        Write-Output 'STOPPED: no matching workflow processes were active.'
        exit 0
    }

    $depth = Get-DepthMap $processes
    $ordered = $processes | Sort-Object @{Expression = { $depth[[int]$_.ProcessId] }; Descending = $true}, CreationDate -Descending

    foreach ($p in $ordered) {
        $pidValue = [int]$p.ProcessId
        if ($PSCmdlet.ShouldProcess("PID $pidValue", 'Stop repository workflow process')) {
            Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
            Write-Output "Stopped PID $pidValue | $($p.Name)"
        }
    }

    Start-Sleep -Seconds 2
    $remaining = @(Get-WorkflowProcesses)
    if ($remaining.Count -gt 0) {
        $ids = ($remaining | ForEach-Object { $_.ProcessId }) -join ', '
        Write-RunStatus -ProcessStatus "Stop incomplete; remaining PIDs: $ids" -ErrorText 'Some workflow processes remain active'
        Write-Error "Workflow stop incomplete. Remaining PIDs: $ids"
        exit 2
    }

    foreach ($lock in $lockPaths) {
        if (Test-Path $lock) {
            Remove-Item $lock -Force
            Write-Output "Removed stale lock: $lock"
        }
    }

    Write-RunStatus -ProcessStatus 'All repository workflow processes stopped' -ErrorText 'None'
    Write-Output 'STOPPED: all matching workflow processes are stopped; results and checkpoints were preserved.'
    exit 0
}
catch {
    try { Write-RunStatus -ProcessStatus 'Stop failed' -ErrorText $_.Exception.Message } catch {}
    Write-Error $_
    exit 1
}
