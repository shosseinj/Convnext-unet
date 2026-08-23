import unittest

from amp_training import add_scaler_state, finish_optimizer_step, restore_scaler_state
from amp_preflight import candidate_batch_sizes


class FakeScaler:
    def __init__(self):
        self.loaded = None
        self.updates = 0
        self.steps = 0

    def state_dict(self):
        return {"scale": 1024.0}

    def load_state_dict(self, state):
        self.loaded = state

    def step(self, _optimizer):
        self.steps += 1

    def update(self):
        self.updates += 1


class FakeOptimizer:
    def __init__(self):
        self.zeroed = 0
        self.steps = 0

    def zero_grad(self, set_to_none=False):
        self.zeroed += int(set_to_none)

    def step(self):
        self.steps += 1


class AmpTrainingTests(unittest.TestCase):
    def test_scaler_state_round_trips_through_checkpoint(self):
        checkpoint = {}
        add_scaler_state(checkpoint, FakeScaler(), enabled=True)
        self.assertEqual(checkpoint["scaler_state_dict"], {"scale": 1024.0})
        restored = FakeScaler()
        self.assertTrue(restore_scaler_state(checkpoint, restored, enabled=True))
        self.assertEqual(restored.loaded, {"scale": 1024.0})

    def test_disabled_amp_does_not_write_or_require_scaler_state(self):
        checkpoint = {}
        add_scaler_state(checkpoint, FakeScaler(), enabled=False)
        self.assertNotIn("scaler_state_dict", checkpoint)
        self.assertFalse(restore_scaler_state(checkpoint, FakeScaler(), enabled=False))

    def test_preflight_fallback_order_preserves_fair_shared_batch_policy(self):
        self.assertEqual(candidate_batch_sizes(24), (24, 20, 16))
        self.assertEqual(candidate_batch_sizes(20), (20, 16))

    def test_amp_overflow_updates_scale_instead_of_getting_stuck(self):
        scaler = FakeScaler()
        optimizer = FakeOptimizer()
        stepped = finish_optimizer_step(
            optimizer, scaler, amp_enabled=True, gradients_are_finite=False
        )
        self.assertFalse(stepped)
        self.assertEqual(scaler.updates, 1)
        self.assertEqual(scaler.steps, 0)
        self.assertEqual(optimizer.zeroed, 1)


if __name__ == "__main__":
    unittest.main()
