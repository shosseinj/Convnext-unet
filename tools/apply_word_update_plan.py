"""Apply a verified Word update plan to a new DOCX using deterministic OOXML."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET

try:
    from .build_word_update_plan import build_plan
    from .check_word_update_readiness import sha256
except ImportError:  # direct-script execution
    from build_word_update_plan import build_plan
    from check_word_update_readiness import sha256


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W = f"{{{W_NS}}}"
LEGACY_TERMS = ("Learned Local-Response Skip Enhancement", "LRS-E", "LRSE")
PLACEHOLDER_RE = re.compile(r"\b(?:XX|TBD)\b|\[(?:AUTHOR NAMES|AFFILIATIONS|XX_[^]]+)\]", re.I)


def _text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(W + "t"))


def _set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    ppr = paragraph.find(W + "pPr")
    first_run = paragraph.find(W + "r")
    rpr = first_run.find(W + "rPr") if first_run is not None else None
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)
    run = ET.SubElement(paragraph, W + "r")
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    node = ET.SubElement(run, W + "t")
    if text[:1].isspace() or text[-1:].isspace():
        node.set(f"{{{XML_NS}}}space", "preserve")
    node.text = text


def _replace_paragraphs(body: ET.Element, plan: dict) -> None:
    paragraphs = body.findall(W + "p")
    for action in sorted(plan["actions"], key=lambda item: item["paragraph"], reverse=True):
        index = action["paragraph"]
        if index >= len(paragraphs):
            raise RuntimeError(f"Paragraph index out of range during apply: {index}")
        paragraph = paragraphs[index]
        position = list(body).index(paragraph)
        if action["action"] == "delete":
            body.remove(paragraph)
            continue
        replacements = action["replacement_paragraphs"]
        if not replacements:
            raise RuntimeError(f"Paragraph {index} has no materialized replacement")
        _set_paragraph_text(paragraph, replacements[0])
        for offset, text in enumerate(replacements[1:], start=1):
            new_paragraph = copy.deepcopy(paragraph)
            _set_paragraph_text(new_paragraph, text)
            body.insert(position + offset, new_paragraph)


def _apply_globals(root: ET.Element, replacements: list[dict]) -> None:
    ordered = sorted(replacements, key=lambda item: len(item["find"]), reverse=True)
    for node in root.iter(W + "t"):
        value = node.text or ""
        for replacement in ordered:
            value = value.replace(replacement["find"], replacement["replace"])
        node.text = value


def _apply_structured(body: ET.Element, updates: list[dict]) -> None:
    tables = body.findall(W + "tbl")
    paragraphs = body.findall(W + "p")
    for update in updates:
        table_index = update["table_index"]
        if table_index >= len(tables):
            raise RuntimeError(f"Table index out of range: {table_index}")
        table = tables[table_index]
        anchor = paragraphs[update["anchor_after_paragraph"]]
        elements = list(body)
        if elements.index(table) != elements.index(anchor) + 1:
            raise RuntimeError(f"Table {table_index} no longer follows its paragraph anchor")
        rows = table.findall(W + "tr")
        for cell_patch in update["cells"]:
            row_index, column_index = cell_patch["row"], cell_patch["column"]
            if row_index >= len(rows):
                raise RuntimeError(f"Table row out of range: {table_index}/{row_index}")
            cells = rows[row_index].findall(W + "tc")
            if column_index >= len(cells):
                raise RuntimeError(f"Table column out of range: {table_index}/{row_index}/{column_index}")
            cell = cells[column_index]
            actual = _text(cell)
            if actual != cell_patch["expected"]:
                raise RuntimeError(
                    f"Table cell drift at {table_index}/{row_index}/{column_index}: "
                    f"expected {cell_patch['expected']!r}, found {actual!r}")
            cell_paragraph = cell.find(W + "p")
            if cell_paragraph is None:
                cell_paragraph = ET.SubElement(cell, W + "p")
            _set_paragraph_text(cell_paragraph, cell_patch["text"])


def _verify_backup(root: Path, source: Path, source_hash: str, manifest_path: Path | None) -> dict:
    if manifest_path is None:
        candidates = sorted((root / "manuscript" / "backups").glob("*.docx.manifest.json"))
        if not candidates:
            raise RuntimeError("No verified pre-edit backup manifest exists")
        manifest_path = candidates[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backup = Path(manifest.get("backup", ""))
    if (manifest.get("status") != "PASS" or manifest.get("readiness_status") != "READY"
            or Path(manifest.get("source", "")).resolve() != source
            or manifest.get("source_sha256") != source_hash
            or not backup.is_file()
            or sha256(backup) != source_hash
            or manifest.get("backup_sha256") != source_hash):
        raise RuntimeError("Pre-edit backup manifest or backup hash is invalid")
    return {"manifest": str(manifest_path), "backup": str(backup), "sha256": source_hash}


def apply_plan(root: Path, output: Path, backup_manifest: Path | None = None) -> dict:
    root = root.resolve(); output = output.resolve()
    plan = build_plan(root)
    source = Path(plan["source_docx"]["path"]).resolve()
    if source == output:
        raise RuntimeError("Refusing to overwrite the source DOCX")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if sha256(source) != plan["source_docx"]["sha256"]:
        raise RuntimeError("Source DOCX hash changed after plan construction")
    backup = _verify_backup(root, source, plan["source_docx"]["sha256"], backup_manifest)

    with ZipFile(source) as archive:
        document_xml = archive.read("word/document.xml")
        root_xml = ET.fromstring(document_xml)
        body = root_xml.find(W + "body")
        if body is None:
            raise RuntimeError("DOCX document.xml has no body")
        _apply_structured(body, plan.get("structured_updates", []))
        _replace_paragraphs(body, plan)
        _apply_globals(root_xml, plan.get("global_replacements", []))
        final_text = _text(root_xml)
        remaining_legacy = [term for term in LEGACY_TERMS if term in final_text]
        remaining_placeholders = sorted(set(PLACEHOLDER_RE.findall(final_text)))
        if remaining_legacy or remaining_placeholders:
            raise RuntimeError(
                f"Post-edit content gate failed; legacy={remaining_legacy}, "
                f"placeholders={remaining_placeholders}")
        updated_xml = ET.tostring(root_xml, encoding="utf-8", xml_declaration=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(suffix=".docx", dir=output.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with ZipFile(temporary, "w", ZIP_DEFLATED) as target:
                for info in archive.infolist():
                    target.writestr(info, updated_xml if info.filename == "word/document.xml" else archive.read(info.filename))
            with ZipFile(temporary) as check:
                bad = check.testzip()
                if bad is not None:
                    raise RuntimeError(f"Output DOCX ZIP integrity failed at {bad}")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"status": "PASS", "output": str(output), "sha256": sha256(output),
            "source_sha256": plan["source_docx"]["sha256"], "backup": backup,
            "actions": plan["action_count"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("manuscript/hossein_paper_revised_final.docx"))
    parser.add_argument("--backup-manifest", type=Path)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    try:
        result = apply_plan(args.root, output, args.backup_manifest)
    except (RuntimeError, FileExistsError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
