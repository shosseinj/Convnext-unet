import tempfile
import unittest
from pathlib import Path

from tools.audit_manuscript_artifacts import audit


class ManuscriptArtifactAuditTests(unittest.TestCase):
    def test_missing_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing or empty artifact"):
                audit(Path(directory))


if __name__ == "__main__":
    unittest.main()
