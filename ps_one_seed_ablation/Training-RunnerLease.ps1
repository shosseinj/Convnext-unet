function Get-TrainingRunnerLeaseOwnerPid {
    param([Parameter(Mandatory = $true)][string] $LeasePath)

    try {
        $content = Get-Content -LiteralPath $LeasePath -Raw -ErrorAction Stop
        if ($content -match '(?m)^owner_pid=(\d+)$') { return [int]$Matches[1] }
    } catch {
        return $null
    }
    return $null
}

function Enter-TrainingRunnerLease {
    param([Parameter(Mandatory = $true)][string] $LeasePath)

    $parent = Split-Path -Parent $LeasePath
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    for ($attempt = 0; $attempt -lt 2; $attempt++) {
        try {
            $stream = [System.IO.File]::Open(
                $LeasePath,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::Read
            )
            $record = "owner_pid=$PID`ncreated_utc=$([DateTime]::UtcNow.ToString('o'))`n"
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($record)
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush()
            return [pscustomobject]@{ Path = $LeasePath; Stream = $stream; OwnerPid = $PID }
        } catch [System.IO.IOException] {
            Start-Sleep -Milliseconds 50
            $ownerPid = Get-TrainingRunnerLeaseOwnerPid -LeasePath $LeasePath
            $owner = if ($ownerPid) { Get-Process -Id $ownerPid -ErrorAction SilentlyContinue } else { $null }
            if ($owner) {
                throw "Another runner owns the training lease (PID $ownerPid)."
            }

            $ageSeconds = if (Test-Path -LiteralPath $LeasePath) {
                ((Get-Date).ToUniversalTime() - (Get-Item -LiteralPath $LeasePath).LastWriteTimeUtc).TotalSeconds
            } else { 0 }
            if ($ageSeconds -lt 30) {
                throw "The training lease is being initialized or was recently interrupted: $LeasePath"
            }
            Remove-Item -LiteralPath $LeasePath -Force -ErrorAction Stop
        }
    }
    throw "Could not acquire the training lease: $LeasePath"
}

function Exit-TrainingRunnerLease {
    param([Parameter(Mandatory = $true)] $Lease)

    try {
        if ($Lease.Stream) { $Lease.Stream.Dispose() }
    } finally {
        if (Test-Path -LiteralPath $Lease.Path) {
            $ownerPid = Get-TrainingRunnerLeaseOwnerPid -LeasePath $Lease.Path
            if ($ownerPid -eq $Lease.OwnerPid) {
                Remove-Item -LiteralPath $Lease.Path -Force -ErrorAction Stop
            }
        }
    }
}
