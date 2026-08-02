import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_ugbr_pilot_campaign import (
    PILOT_JOBS, PILOT_NAMESPACE, SEED, STATUS_FIELDS, acquire_lock,
    metadata_equivalent, pilot_run_dir, run_campaign, training_command,
    validation_command, write_status,
)


class PilotCampaignTests(unittest.TestCase):
    def test_fixed_order_and_seed(self):
        self.assertEqual(PILOT_JOBS, (
            "baseline", "baseline_ugbr", "baseline_best_existing",
            "baseline_best_existing_ugbr",
        ))
        self.assertEqual(SEED, 42)

    def test_command_constructs_pilot_and_optional_resume(self):
        command = training_command("target-python", Path("repo"), "baseline_ugbr", "cuda", False)
        self.assertEqual(command[0:2], ["target-python", "-u"])
        self.assertIn("--pilot", command)
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertNotIn("--resume", command)
        resumed = training_command("target-python", Path("repo"), "baseline_ugbr", "cuda", True)
        self.assertEqual(resumed[-1], "--resume")

    def test_training_targets_only_fresh_pilot_namespace(self):
        command = training_command("python", Path("repo"), "baseline", "cuda")
        output_root = Path(command[command.index("--output-root") + 1])
        self.assertEqual(output_root, Path("repo") / PILOT_NAMESPACE)
        self.assertEqual(pilot_run_dir(Path("repo"), "baseline"),
                         Path("repo") / PILOT_NAMESPACE / "baseline" / "seed_42" / "pilot")
        self.assertNotIn("results/raw", str(output_root))

    def test_legacy_raw_artifacts_cannot_be_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "results/raw/baseline/seed_42/pilot"
            legacy.mkdir(parents=True)
            (legacy / "last.pth").write_bytes(b"legacy")
            self.assertEqual(pilot_run_dir(root, "baseline"),
                             root / PILOT_NAMESPACE / "baseline/seed_42/pilot")
            self.assertFalse(pilot_run_dir(root, "baseline").exists())
            self.assertEqual((legacy / "last.pth").read_bytes(), b"legacy")

    def test_validation_is_an_independent_subprocess_command(self):
        command = validation_command("target-python", Path("validator.py"), Path("repo"), "baseline")
        self.assertEqual(command[:2], ["target-python", "-u"])
        self.assertEqual(command[3:5], ["--validate-run", "baseline"])

    def test_protocol_equivalence_gate_reports_exact_mismatch(self):
        expected = {"variant": "baseline", "seed": 42, "protocol": {"input_size": [352, 352]}}
        self.assertEqual(metadata_equivalent(dict(expected), expected), (True, []))
        changed = dict(expected, seed=3407)
        self.assertEqual(metadata_equivalent(changed, expected), (False, ["seed"]))

    def test_stop_on_failure_shape(self):
        class Result:
            returncode = 9
        calls = []
        def fake_run(command, cwd=None):
            calls.append(command)
            return Result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / ".agentic/ugbr_pilot_campaign.lock.json"
            lock.parent.mkdir(parents=True)
            owned = {"pid": os.getpid()}
            lock.write_text(json.dumps(owned), encoding="utf-8")
            with patch("tools.run_ugbr_pilot_campaign.acquire_lock", return_value=owned), \
                 patch("tools.run_ugbr_pilot_campaign.validate_run", return_value=(False, "missing")), \
                 patch("tools.run_ugbr_pilot_campaign.resume_compatible", return_value=(False, "fresh")):
                with self.assertRaisesRegex(RuntimeError, "training failed for baseline"):
                    run_campaign(root, "target-python", "cuda", run=fake_run)
        self.assertEqual(len(calls), 1)
        self.assertIn("baseline", calls[0])

    def test_lock_rejects_live_pid_and_replaces_only_stale_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "campaign.lock.json"
            lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                acquire_lock(lock)
            lock.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
            with patch("tools.run_ugbr_pilot_campaign.process_alive", return_value=False):
                owned = acquire_lock(lock)
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8")), owned)

    def test_status_has_only_concise_contract_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_status(root, workflow_status="RUNNING", active_run="baseline", completed=0,
                         validation="pending", process_status="active", error="None", next_action="validate")
            lines = (root / "RUN_STATUS.md").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(STATUS_FIELDS))
            self.assertEqual([line.split(":", 1)[0] for line in lines], list(STATUS_FIELDS))
            self.assertLessEqual(len(lines), 15)


if __name__ == "__main__":
    unittest.main()
