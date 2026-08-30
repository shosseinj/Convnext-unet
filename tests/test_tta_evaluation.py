import torch
import json
import subprocess
from pathlib import Path

from evaluation_core import tta_probability


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "ps_one_seed_ablation" / "Evaluate-39-FAFEM-MSCB-Detail-TTA.ps1"


class ConstantLogitModel(torch.nn.Module):
    def forward(self, image):
        return torch.full_like(image[:, :1], 0.75)


def test_tta_probability_preserves_shape_and_averages_inverted_transforms():
    image = torch.randn(1, 3, 32, 32)
    probability = tta_probability(ConstantLogitModel().eval(), image)

    assert probability.shape == (1, 1, 32, 32)
    assert torch.isfinite(probability).all()
    assert probability.min() >= 0
    assert probability.max() <= 1
    assert torch.allclose(probability[..., 16, 16], torch.sigmoid(torch.tensor([[[0.75]]])), atol=1e-6)


def test_exp39_tta_runner_writes_only_separate_tta_summaries():
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(RUNNER), "-DryRun"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(payloads) == 3
    for seed, payload in zip((42, 6543, 7777), payloads):
        command = payload["evaluate"]
        assert command[command.index("--experiment_name") + 1] == "one_seed_39_fafem_mscb_lite_detail_warmup_cosine"
        assert command[command.index("--seed") + 1] == str(seed)
        assert "--tta" in command
        assert command[command.index("--output") + 1].endswith("evaluation_summary_tta.json")
