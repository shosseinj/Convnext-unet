import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

from tools.apply_word_update_plan import W, apply_plan
from tools.check_word_update_readiness import AUTHOR_FIELDS, sha256


def paragraph(text):
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


class ApplyWordUpdatePlanTests(unittest.TestCase):
    def fixture(self):
        root = Path(tempfile.mkdtemp())
        (root / "reports").mkdir(); (root / "manuscript").mkdir()
        source = root / "source.docx"
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            + paragraph("Old title") + paragraph("Old section")
            + '<w:tbl><w:tr><w:tc>' + paragraph("XX") + '</w:tc></w:tr></w:tbl>'
            + paragraph("Delete me") + paragraph("OldTerm remains here")
            + '<w:sectPr/></w:body></w:document>')
        with ZipFile(source, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>")
            archive.writestr("word/document.xml", document)
        section = root / "manuscript" / "section.md"
        section.write_text("# Heading\n\nFirst paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
        target_map = {
            "source": str(source), "source_sha256": sha256(source), "body_paragraph_count": 4,
            "global_replacements": [{"find": "OldTerm", "replace": "BSEI"}],
            "structured_updates": [{"kind": "table_cell_patch", "table_index": 0,
                "anchor_after_paragraph": 1,
                "cells": [{"row": 0, "column": 0, "expected": "XX", "text": "96"}]}],
            "targets": [
                {"paragraph": 0, "action": "replace", "status": "READY_FOR_WORD", "replacement_text": "New title"},
                {"paragraph": 1, "action": "replace", "status": "READY_FOR_WORD",
                 "replacement_source": "manuscript/section.md", "replacement_mode": "markdown_body"},
                {"paragraph": 2, "action": "delete", "status": "READY_FOR_WORD"},
            ]}
        (root / "reports" / "docx_target_paragraphs.json").write_text(json.dumps(target_map), encoding="utf-8")
        (root / "reports" / "manuscript_artifact_gate.json").write_text(json.dumps({
            "status": "PASS", "numerical_results": "READY", "qualitative_package": "PASS"}), encoding="utf-8")
        (root / "manuscript" / "author_inputs.json").write_text(
            json.dumps({field: "provided" for field in AUTHOR_FIELDS}), encoding="utf-8")
        backup_dir = root / "manuscript" / "backups"; backup_dir.mkdir()
        backup = backup_dir / "source.before_word_update.test.docx"
        backup.write_bytes(source.read_bytes())
        manifest = backup.with_suffix(".docx.manifest.json")
        manifest.write_text(json.dumps({
            "status": "PASS", "readiness_status": "READY", "source": str(source),
            "source_sha256": sha256(source), "backup": str(backup),
            "backup_sha256": sha256(backup)}), encoding="utf-8")
        return root, source

    def test_applies_to_new_docx_and_preserves_source(self):
        root, source = self.fixture(); source_hash = sha256(source)
        output = root / "final.docx"
        result = apply_plan(root, output)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(sha256(source), source_hash)
        with ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            xml = ET.fromstring(archive.read("word/document.xml"))
        body = xml.find(W + "body")
        texts = ["".join(t.text or "" for t in p.iter(W + "t")) for p in body.findall(W + "p")]
        self.assertEqual(texts, ["New title", "First paragraph.", "Second paragraph.", "BSEI remains here"])
        self.assertEqual("".join(t.text or "" for t in body.find(W + "tbl").iter(W + "t")), "96")

    def test_refuses_source_overwrite(self):
        root, source = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "overwrite the source"):
            apply_plan(root, source)

    def test_refuses_to_edit_without_verified_backup(self):
        root, source = self.fixture()
        for path in (root / "manuscript" / "backups").iterdir():
            path.unlink()
        with self.assertRaisesRegex(RuntimeError, "backup manifest"):
            apply_plan(root, root / "final.docx")


if __name__ == "__main__":
    unittest.main()
