import json
import tempfile
import unittest
from pathlib import Path

from tools.build_word_update_plan import build_plan
from tools.check_word_update_readiness import AUTHOR_FIELDS, sha256


class WordUpdatePlanTests(unittest.TestCase):
    def fixture(self, ready=True):
        root = Path(tempfile.mkdtemp())
        (root / "reports").mkdir(); (root / "manuscript").mkdir()
        source = root / "source.docx"; source.write_bytes(b"docx")
        section = root / "manuscript" / "section.md"
        section.write_text("# Heading\n\nFirst line\ncontinues.\n\nEquation block.\n", encoding="utf-8")
        targets = [
            {"paragraph": 0, "action": "replace", "status": "READY_FOR_WORD",
             "replacement_text": "Title"},
            {"paragraph": 1, "action": "replace", "status": "READY_FOR_WORD",
             "replacement_source": "manuscript/section.md", "replacement_mode": "markdown_body"},
            {"paragraph": 2, "action": "delete", "status": "READY_FOR_WORD"},
        ]
        if not ready:
            targets[0]["status"] = "BLOCKED_BY_EXPERIMENT"
        (root / "reports" / "docx_target_paragraphs.json").write_text(json.dumps({
            "source": str(source), "source_sha256": sha256(source),
            "body_paragraph_count": 3,
            "global_replacements": [{"find": "OldTerm", "replace": "BSEI"}],
            "structured_updates": [{"kind": "table_cell_patch", "table_index": 0,
                "anchor_after_paragraph": 1, "cells": [{"row": 0, "column": 0,
                    "expected": "OldTerm", "text": "BSEI"}]}],
            "targets": targets}), encoding="utf-8")
        (root / "reports" / "manuscript_artifact_gate.json").write_text(json.dumps({
            "status": "PASS", "numerical_results": "READY", "qualitative_package": "PASS"}),
            encoding="utf-8")
        (root / "manuscript" / "author_inputs.json").write_text(
            json.dumps({field: "provided" for field in AUTHOR_FIELDS}), encoding="utf-8")
        return root

    def test_materializes_hash_bound_actions_without_heading(self):
        plan = build_plan(self.fixture())
        self.assertEqual(plan["status"], "PASS")
        self.assertEqual(plan["action_count"], 3)
        self.assertEqual(plan["actions"][1]["replacement_paragraphs"],
                         ["First line continues.", "Equation block."])
        self.assertEqual(len(plan["actions"][1]["replacement_source_sha256"]), 64)
        self.assertEqual(plan["actions"][2]["replacement_paragraphs"], [])
        self.assertEqual(plan["global_replacements"][0]["replace"], "BSEI")
        self.assertEqual(plan["structured_updates"][0]["cells"][0]["text"], "BSEI")

    def test_refuses_to_build_while_readiness_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "UNRESOLVED_DOCX_TARGETS"):
            build_plan(self.fixture(False))


if __name__ == "__main__":
    unittest.main()
