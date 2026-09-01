import sys
import unittest

import main_torch


class ProgressBarOutputTests(unittest.TestCase):
    def test_progress_bars_write_to_stderr_not_training_log_stdout(self):
        progress = main_torch.make_progress_bar([], desc="test")
        try:
            self.assertIs(getattr(progress.fp, "_wrapped", progress.fp), sys.stderr)
        finally:
            progress.close()


if __name__ == "__main__":
    unittest.main()
