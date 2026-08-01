import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.prepare_word_update_backup import create_backup


class WordUpdateBackupTests(unittest.TestCase):
    def test_blocked_gate_creates_no_backup(self):
        root = Path(tempfile.mkdtemp())
        output = root / "backups"
        with patch("tools.prepare_word_update_backup.assess_readiness", return_value={
            "status": "BLOCKED", "blockers": [{"id": "NUMERICAL_RESULTS_NOT_READY"}]
        }):
            with self.assertRaises(RuntimeError):
                create_backup(root, output, "20260801T000000Z")
        self.assertFalse(output.exists())

    def test_ready_gate_creates_hash_identical_backup_and_manifest(self):
        root = Path(tempfile.mkdtemp())
        source = root / "article.docx"
        source.write_bytes(b"word-source")
        import hashlib
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        readiness = {
            "status": "READY", "blockers": [],
            "source": {"path": str(source), "sha256": digest},
        }
        with patch("tools.prepare_word_update_backup.assess_readiness", return_value=readiness):
            result = create_backup(root, Path("backups"), "20260801T000000Z")
        backup = Path(result["backup"])
        self.assertEqual(backup.read_bytes(), source.read_bytes())
        manifest = json.loads(backup.with_suffix(".docx.manifest.json").read_text())
        self.assertEqual(manifest["source_sha256"], manifest["backup_sha256"])

    def test_existing_backup_is_never_overwritten(self):
        root = Path(tempfile.mkdtemp())
        source = root / "article.docx"
        source.write_bytes(b"word-source")
        import hashlib
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        readiness = {
            "status": "READY", "blockers": [],
            "source": {"path": str(source), "sha256": digest},
        }
        with patch("tools.prepare_word_update_backup.assess_readiness", return_value=readiness):
            create_backup(root, Path("backups"), "20260801T000000Z")
            with self.assertRaises(FileExistsError):
                create_backup(root, Path("backups"), "20260801T000000Z")


if __name__ == "__main__":
    unittest.main()
