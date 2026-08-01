import json
import tempfile
import unittest
from pathlib import Path

from tools.check_word_update_readiness import AUTHOR_FIELDS, assess_readiness, sha256


class WordUpdateReadinessTests(unittest.TestCase):
    def fixture(self, ready: bool) -> Path:
        root = Path(tempfile.mkdtemp())
        (root / "reports").mkdir()
        (root / "manuscript").mkdir()
        source = root / "source.docx"
        source.write_bytes(b"docx-fixture")
        status = "READY_FOR_WORD" if ready else "BLOCKED_BY_EXPERIMENT"
        (root / "reports" / "docx_target_paragraphs.json").write_text(json.dumps({
            "source": str(source),
            "source_sha256": sha256(source),
            "body_paragraph_count": 10,
            "targets": [{
                "paragraph": 5,
                "action": "replace",
                "status": status,
                "replacement_text": "verified final text" if ready else None,
            }],
        }), encoding="utf-8")
        (root / "reports" / "manuscript_artifact_gate.json").write_text(json.dumps({
            "status": "PASS",
            "numerical_results": "READY" if ready else "BLOCKED_BY_EXPERIMENT",
            "qualitative_package": "PASS" if ready else "BLOCKED_BY_EVALUATION",
        }), encoding="utf-8")
        if ready:
            (root / "manuscript" / "author_inputs.json").write_text(
                json.dumps({field: "provided" for field in AUTHOR_FIELDS}), encoding="utf-8"
            )
        return root

    def test_ready_only_when_every_contract_is_satisfied(self):
        report = assess_readiness(self.fixture(True))
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["blockers"], [])

    def test_fail_closed_for_experiment_and_author_blockers(self):
        report = assess_readiness(self.fixture(False))
        self.assertEqual(report["status"], "BLOCKED")
        ids = {item["id"] for item in report["blockers"]}
        self.assertIn("UNRESOLVED_DOCX_TARGETS", ids)
        self.assertIn("NUMERICAL_RESULTS_NOT_READY", ids)
        self.assertIn("QUALITATIVE_PACKAGE_NOT_READY", ids)
        self.assertIn("AUTHOR_INPUTS_MISSING", ids)

    def test_source_hash_mismatch_blocks_update(self):
        root = self.fixture(True)
        (root / "source.docx").write_bytes(b"changed")
        report = assess_readiness(root)
        self.assertIn("SOURCE_DOCX_HASH_MISMATCH", {b["id"] for b in report["blockers"]})

    def test_nested_empty_author_template_does_not_pass(self):
        root = self.fixture(True)
        (root / "manuscript" / "author_inputs.json").write_text(json.dumps({
            "authors": [{"name": "", "affiliation_ids": []}],
            "affiliations": [{"id": "", "text": ""}],
            "corresponding_author": {"name": "", "email": ""},
            "code_availability": {"url": "", "release_conditions": ""},
            "credit_roles": [{"author": "", "roles": []}],
        }), encoding="utf-8")
        report = assess_readiness(root)
        blocker = next(b for b in report["blockers"] if b["id"] == "AUTHOR_INPUTS_INCOMPLETE")
        self.assertEqual(set(blocker["fields"]), set(AUTHOR_FIELDS))

    def test_ready_replacement_requires_exactly_one_existing_payload(self):
        root = self.fixture(True)
        target_path = root / "reports" / "docx_target_paragraphs.json"
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        payload["targets"][0].pop("replacement_text")
        payload["targets"][0]["replacement_source"] = "manuscript/missing.md"
        payload["targets"][0]["replacement_mode"] = "markdown_body"
        target_path.write_text(json.dumps(payload), encoding="utf-8")
        report = assess_readiness(root)
        blocker = next(b for b in report["blockers"] if b["id"] == "INVALID_DOCX_TARGETS")
        self.assertIn("replacement_source is missing", blocker["targets"][0]["errors"][0])

    def test_duplicate_out_of_range_and_unknown_actions_fail_closed(self):
        root = self.fixture(True)
        target_path = root / "reports" / "docx_target_paragraphs.json"
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        payload["targets"] = [
            {"paragraph": 10, "action": "merge", "status": "READY_FOR_WORD"},
            {"paragraph": 10, "action": "delete", "status": "READY_FOR_WORD"},
        ]
        target_path.write_text(json.dumps(payload), encoding="utf-8")
        report = assess_readiness(root)
        blocker = next(b for b in report["blockers"] if b["id"] == "INVALID_DOCX_TARGETS")
        errors = [error for target in blocker["targets"] for error in target["errors"]]
        self.assertTrue(any("outside body_paragraph_count" in error for error in errors))
        self.assertTrue(any("duplicated" in error for error in errors))
        self.assertTrue(any("action must" in error for error in errors))

    def test_source_replacement_requires_explicit_materialization_mode(self):
        root = self.fixture(True)
        source = root / "manuscript" / "section.md"
        source.write_text("# Heading\n\nFinal text.\n", encoding="utf-8")
        target_path = root / "reports" / "docx_target_paragraphs.json"
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        payload["targets"][0].pop("replacement_text")
        payload["targets"][0]["replacement_source"] = "manuscript/section.md"
        target_path.write_text(json.dumps(payload), encoding="utf-8")
        report = assess_readiness(root)
        blocker = next(b for b in report["blockers"] if b["id"] == "INVALID_DOCX_TARGETS")
        self.assertIn("requires markdown_body", blocker["targets"][0]["errors"][0])

    def test_invalid_structured_and_global_updates_fail_closed(self):
        root = self.fixture(True)
        target_path = root / "reports" / "docx_target_paragraphs.json"
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        payload["structured_updates"] = [{"kind": "unknown", "table_index": -1, "cells": []}]
        payload["global_replacements"] = [{"find": "", "replace": ""}]
        target_path.write_text(json.dumps(payload), encoding="utf-8")
        ids = {item["id"] for item in assess_readiness(root)["blockers"]}
        self.assertIn("INVALID_STRUCTURED_UPDATES", ids)
        self.assertIn("INVALID_GLOBAL_REPLACEMENTS", ids)


if __name__ == "__main__":
    unittest.main()
