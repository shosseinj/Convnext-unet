import random
import unittest

import numpy as np
import torch

from train_research import (capture_rng_state, restore_rng_state, update_early_stopping,
                            validate_resume_metadata)


class RecoveryMetadataTests(unittest.TestCase):
    def test_rng_state_round_trip(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        generator = torch.Generator().manual_seed(123)
        state = capture_rng_state(generator)

        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.rand(3, generator=generator),
        )
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        generator.manual_seed(999)
        restore_rng_state(state, generator)
        actual = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.rand(3, generator=generator),
        )
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_resume_metadata_rejects_mismatch_and_legacy_checkpoint(self):
        current = {
            "variant": "baseline",
            "seed": 42,
            "run_mode": "official",
            "manifest_sha256": "manifest",
            "protocol": {"input_size": [352, 352]},
            "source_fingerprints": {"train_research.py": {"sha256": "code"}},
            "pretrained_weights": {"sha256": "weights"},
        }
        validate_resume_metadata(dict(current), current)
        changed = dict(current, seed=3407)
        with self.assertRaisesRegex(RuntimeError, "immutable metadata mismatch"):
            validate_resume_metadata(changed, current)
        legacy = {"variant": "baseline", "manifest_sha256": "manifest"}
        with self.assertRaisesRegex(RuntimeError, "predates reproducible recovery contract"):
            validate_resume_metadata(legacy, current)

    def test_early_stopping_patience_starts_after_unfreeze(self):
        self.assertEqual(update_early_stopping(0.4, 0.5, 0, True), (0.5, 0, False))
        self.assertEqual(update_early_stopping(0.4, 0.5, 0, False), (0.5, 1, False))
        self.assertEqual(update_early_stopping(0.6, 0.5, 7, False), (0.6, 0, True))


if __name__ == "__main__":
    unittest.main()
