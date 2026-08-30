import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "ps_one_seed_ablation" / "Training-RunnerLease.ps1"


def test_runner_lease_blocks_a_second_contender(tmp_path):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    assert powershell, "PowerShell is required to test the runner lease"
    lease_path = tmp_path / "training.lock"
    command = f'''\
. '{HELPER}'
$lease = Enter-TrainingRunnerLease -LeasePath '{lease_path}'
try {{
    $blocked = $false
    try {{ Enter-TrainingRunnerLease -LeasePath '{lease_path}' | Out-Null }} catch {{ $blocked = $true }}
    if (-not $blocked) {{ throw 'second contender acquired the lease' }}
}} finally {{
    Exit-TrainingRunnerLease -Lease $lease
}}
'''
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
