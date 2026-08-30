import json
import subprocess
import unittest
from pathlib import Path

from ablation_registry import get_experiment


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "one_seed_37_fafem_mscb_lite_stage3_warmup_cosine"
TARGET = "one_seed_38_fafem_mscb_lite_stage3_cosine_refinement"
RUNNER = ROOT / "ps_one_seed_ablation" / "38_fafem_mscb_lite_stage3_cosine_refinement.ps1"


class FAFEMMSCBRefinementProtocolTests(unittest.TestCase):
    def test_target_registry_preserves_the_source_architecture(self):
        source = get_experiment(SOURCE).to_dict()
        target = get_experiment(TARGET).to_dict()
        self.assertEqual(source.pop("name"), SOURCE)
        self.assertEqual(target.pop("name"), TARGET)
        self.assertEqual(source, target)

    def test_runner_dry_run_uses_each_seed_37_best_checkpoint(self):
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-File", str(RUNNER), "-DryRun"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payloads = [
            json.loads(line)
            for line in completed.stdout.splitlines()
            if line.startswith("{")
        ]
        self.assertEqual([payload["experiment"] for payload in payloads], [TARGET] * 3)
        self.assertEqual([payload["seed_dir"].split("seed_")[-1] for payload in payloads], ["42", "6543", "7777"])
        for payload in payloads:
            train = payload["train"]
            self.assertEqual(train[train.index("--enable_fafem") + 1], "True")
            self.assertEqual(train[train.index("--enable_mscb_lite_stage3") + 1], "True")
            self.assertEqual(train[train.index("--lr_scheduler") + 1], "cosine_warm_restarts")
            self.assertEqual(train[train.index("--refinement_force_all_trainable") + 1], "True")
            source = train[train.index("--refinement_checkpoint_path") + 1]
            self.assertIn("37_fafem_mscb_lite_stage3_warmup_cosine", source)
            self.assertTrue(source.endswith("best_checkpoint.pth"))


if __name__ == "__main__":
    unittest.main()
