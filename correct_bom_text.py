"""Apply source-PDF-verified OCR corrections to the checked sample DWG."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from uuid import uuid4

from blockify_pdf_symbols import CORE_CONSOLE, audit, run_console
from bom_ocr_to_dwg import (OCR_JSON, TEXT_LAYER, VERIFIED_TEXT,
                            accepted_cells, cell_bounds, literal)


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "verified_dwg/20261014"
SOURCE = FOLDER / "symbols_59_text_corrected.dwg"
OUTPUT = FOLDER / "symbols_59_text_checked.dwg"
ORIGINAL_MANIFEST = ROOT / "verified_dwg/20261013/symbols_59_with_bom_ocr_v3.csv"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (SOURCE, OCR_JSON, ORIGINAL_MANIFEST, CORE_CONSOLE):
        if not path.is_file():
            raise FileNotFoundError(path)
    shutil.copy2(SOURCE, OUTPUT)
    audit_csv = audit(OUTPUT, FOLDER, f"bom_correction_before_{uuid4().hex[:8]}")
    with audit_csv.open(encoding="utf-8", newline="") as stream:
        entities = list(csv.DictReader(stream))
    with ORIGINAL_MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        original = {(int(row["row"]), int(row["column"])): row
                    for row in csv.DictReader(stream) if row["row"] != "header"}
    cells = accepted_cells(json.loads(OCR_JSON.read_text(encoding="utf-8"))["cells"])
    by_key = {(cell["row"], cell["column"]): cell for cell in cells}

    commands = ['(setq bom_fixed 0)']
    changes = []
    handles: set[str] = set()
    for key, value in VERIFIED_TEXT.items():
        cell = by_key[key]
        old = original[key]
        if not cell["accepted"] or old["accepted"] != "True":
            raise AssertionError(f"BOM cell is not active: {key}")
        if cell["text_to_insert"] != value or old["text_to_insert"] == value:
            raise AssertionError(f"BOM correction changed unexpectedly: {key}")
        x0, _, _, y1 = cell_bounds(cell)
        x, y = x0 + .035, y1 - .045
        matches = [entity for entity in entities
                   if entity["type"] == "MTEXT"
                   and entity["layer"] == TEXT_LAYER
                   and abs(float(entity["cx"]) - x) < .0001
                   and abs(float(entity["cy"]) - y) < .0001]
        if len(matches) != 1:
            raise AssertionError(f"BOM MTEXT not uniquely located: {key}: {len(matches)}")
        handle = matches[0]["handle"]
        if handle in handles:
            raise AssertionError(f"BOM MTEXT selected twice: {handle}")
        handles.add(handle)
        commands.append(
            f'(if (setq bom_ent (handent "{handle}")) '
            '(progn (setq bom_data (entget bom_ent)) '
            f'(if (entmod (subst (cons 1 "{literal(value)}") '
            '(assoc 1 bom_data) bom_data)) '
            '(progn (entupd bom_ent) (setq bom_fixed (1+ bom_fixed))))))'
        )
        changes.append({"row": key[0], "column": key[1], "handle": handle,
                        "old_ocr": old["text_to_insert"], "pdf_verified": value})
    commands += [
        '(princ (strcat "\\nBOM_FIXED=" (itoa bom_fixed)))',
        '_.QSAVE', '_.QUIT', '_N',
    ]
    script = OUTPUT.with_suffix('.scr')
    script.write_text('\n'.join(commands) + '\n', encoding='ascii')
    log = OUTPUT.with_suffix('.log')
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode('utf-16-le', errors='replace')
    if f"BOM_FIXED={len(changes)}" not in report:
        raise RuntimeError(f"BOM text edit incomplete: {log}")

    # Reopen the saved file to verify actual text contents and layer visibility.
    checks = ['(setq bom_ok 0)']
    for change in changes:
        checks.append(
            '(if (equal (cdr (assoc 1 '
            f'(entget (handent "{change["handle"]}")))) '
            f'"{literal(change["pdf_verified"])}") '
            '(setq bom_ok (1+ bom_ok)))'
        )
    checks += [
        '(princ (strcat "\\nBOM_VERIFIED=" (itoa bom_ok)))',
        '_.QUIT', '_N',
    ]
    check_script = OUTPUT.with_name('verify_bom_corrections.scr')
    check_script.write_text('\n'.join(checks) + '\n', encoding='ascii')
    check_log = OUTPUT.with_name('verify_bom_corrections.log')
    run_console(OUTPUT, check_script, check_log)
    verified = check_log.read_bytes().decode('utf-16-le', errors='replace')
    if f"BOM_VERIFIED={len(changes)}" not in verified:
        raise AssertionError(f"Saved BOM MTEXT differs from PDF check: {check_log}")

    after = audit(OUTPUT, FOLDER, f"bom_correction_after_{uuid4().hex[:8]}")
    with after.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_counts = {"BOM_OCR_TEXT": 155, "CAPTION_LLM_TEXT": 31,
                       "WORK_NOTE_MTEXT": 5, "ADDITIONAL_EDITABLE_TEXT": 13}
    for layer, count in expected_counts.items():
        if sum(r["type"] == "MTEXT" and r["layer"] == layer for r in rows) != count:
            raise AssertionError(f"MTEXT count changed on {layer}")
    if sum(r["type"] == "INSERT" for r in rows) != 59:
        raise AssertionError("Symbol blocks changed")
    OUTPUT.with_suffix('.json').write_text(json.dumps(
        {"source": str(SOURCE), "verified_bom_corrections": changes,
         "count": len(changes), "method": "enlarged source-PDF cells matched to DWG MTEXT"},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUTPUT)
    print(f"BOM_FIXED={len(changes)}")


if __name__ == '__main__':
    main()
