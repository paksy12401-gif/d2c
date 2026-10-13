"""Convert BOM-checked PDF outline captions to editable DWG MTEXT.

The `DECISIONS` table is the constrained LLM adjudication: each converted
line was read from a high-resolution crop and its part number checked against
the BOM. Uncertain lines remain as their original vector outlines.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from blockify_pdf_symbols import CORE_CONSOLE, audit, run_console
from bom_ocr_to_dwg import literal


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "verified_dwg/20261013/symbols_59_with_bom_ocr_v3.dwg"
OUTPUT = ROOT / "verified_dwg/20261013/symbols_59_with_bom_and_captions_llm_v4.dwg"
REGIONS = ROOT / "blockified_dwg/caption_crops/regions.json"
BOM = ROOT / "blockified_dwg/bom_cell_ocr.json"
PDF_HEIGHT = 2383.93994140625

# (crop, 1-based line, text, BOM row, top y in PDF points, next-line top)
# The PDF locations are distinct; repeated part numbers at separate locations
# intentionally create separate MTEXT entities.
DECISIONS = [
    (1, 1, "부품명 : 13519739_Y626LEFT_FRONT_MAO_X1", 25, 282.9, 293.4),
    (1, 2, "CONNECTOR NO : 13590001", 24, 293.5, 304.0),
    (2, 1, "부품명 : 19151432_B63DOOR_LFR_X1", 23, 787.9, 798.5),
    (2, 2, "CONNECTOR NO : 19151432", 23, 798.6, 809.0),
    (3, 1, "그로멧 NO : BSB04473 (부자재 NO 16번)", 16, 916.1, 928.0),
    (5, 1, "부품명 : 13551224_X11X500_FL_X1", 22, 973.9, 984.4),
    (5, 2, "CONNECTOR NO : 13551224", 22, 984.5, 995.0),
    (6, 1, "부품명 : 13591047_Y626LEFT_FRONT_MAO_X2", 26, 1067.5, 1078.1),
    (6, 2, "CONNECTOR NO : 13591047", 26, 1078.2, 1089.0),
    (8, 1, "부품명 : 13514590_X604X604_A471_X1", 20, 1312.9, 1323.4),
    (8, 2, "CONNECTOR NO : 13514590", 20, 1323.5, 1334.1),
    (8, 3, "REAR COVER NO : BLL15914", 6, 1334.2, 1345.0),
    (9, 1, "부품명 : 13519739_Y626LEFT_FRONT_MAO_X1", 25, 1371.1, 1381.7),
    (9, 2, "CONNECTOR NO : 13519739", 25, 1381.8, 1392.0),
    (10, 1, "부품명 : 13535680_X644X505_FL_X1", 29, 1394.1, 1404.6),
    (10, 2, "CONNECTOR NO : 13535680", 29, 1404.7, 1415.0),
    (14, 1, "부품명 : 13592509_A470LEFT_FRONT_X1", 15, 2126.7, 2137.3),
    (14, 2, "CONNECTOR NO : 13592509", 15, 2137.4, 2148.0),
    (15, 1, "부품명 : 13532124_X604X604_E603_LT_X1_X1", 19, 2127.4, 2138.0),
    (15, 2, "CONNECTOR NO : 13532124", 19, 2138.1, 2148.6),
    (15, 3, "FSG CLIP NO (28번) : 13352244", 28, 2148.7, 2161.0),
    (16, 1, "부품명 : 13582917_B198LEFT_FRONT_X1", 8, 2127.9, 2138.4),
    (16, 2, "CONNECTOR NO : 13582917", 8, 2138.5, 2149.0),
]


def build() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not SOURCE.is_file() or not CORE_CONSOLE.is_file():
        raise FileNotFoundError("Source DWG or AutoCAD Core Console missing")
    regions = {r["id"]: r for r in json.loads(REGIONS.read_text(encoding="utf-8"))}
    bom_cells = json.loads(BOM.read_text(encoding="utf-8"))["cells"]
    bom_ids = {c["row"]: c["text"].strip() for c in bom_cells if c["column"] == 3}
    for crop, line, value, row, *_ in DECISIONS:
        assert bom_ids[row] in value, (crop, line, row, value, bom_ids[row])

    shutil.copy2(SOURCE, OUTPUT)
    audit_csv = audit(OUTPUT, OUTPUT.parent, "captions_v4_before")
    entities = list(csv.DictReader(audit_csv.open(encoding="utf-8", newline="")))
    commands = [
        '(entmake (list (cons 0 "LAYER") (cons 2 "CAPTION_ORIGINAL_OUTLINES") '
        '(cons 70 0) (cons 62 -8) (cons 6 "Continuous")))',
        '(entmake (list (cons 0 "LAYER") (cons 2 "CAPTION_LLM_TEXT") '
        '(cons 70 0) (cons 62 7) (cons 6 "Continuous")))',
        '(setq cap_moved 0 cap_added 0)',
    ]
    manifest = []
    used_handles = set()
    for crop, line, value, row, top, bottom in DECISIONS:
        rx0, _, rx1, _ = regions[crop]["region_pdf"]
        handles = []
        glyph_boxes = []
        for entity in entities:
            if entity["type"] != "LWPOLYLINE" or entity["layer"] != "PDF_Geomertry":
                continue
            try:
                x0, y0, x1, y1 = (float(entity[k]) for k in
                                  ("minx", "miny", "maxx", "maxy"))
            except ValueError:
                continue
            a, b, c, d = x0 * 72, PDF_HEIGHT - y1 * 72, x1 * 72, PDF_HEIGHT - y0 * 72
            if rx0 - 1 <= a <= c <= rx1 + 1 and top - 0.3 <= b < bottom - 0.3:
                handle = entity["handle"]
                if handle in used_handles:
                    raise AssertionError(f"Glyph assigned twice: {handle}")
                handles.append(handle)
                glyph_boxes.append((a, b, c, d))
                used_handles.add(handle)
        if len(handles) < 10:
            raise AssertionError(f"Too few outlines for crop {crop}, line {line}: {len(handles)}")
        for handle in handles:
            commands.append(
                f'(if (setq cap_ent (handent "{handle}")) '
                f'(progn (setq cap_data (entget cap_ent)) '
                f'(if (entmod (subst (cons 8 "CAPTION_ORIGINAL_OUTLINES") '
                f'(assoc 8 cap_data) cap_data)) (setq cap_moved (1+ cap_moved)))))'
            )
        left = min(b[0] for b in glyph_boxes)
        right = max(b[2] for b in glyph_boxes)
        glyph_top = min(b[1] for b in glyph_boxes)
        x, y = left / 72, (PDF_HEIGHT - glyph_top) / 72
        width = (right - left + 12) / 72
        commands.append(
            f'(if (entmakex (list (cons 0 "MTEXT") (cons 100 "AcDbEntity") '
            f'(cons 8 "CAPTION_LLM_TEXT") (cons 100 "AcDbMText") '
            f'(cons 10 (list {x:.8f} {y:.8f} 0.0)) (cons 40 0.130000) '
            f'(cons 41 {width:.8f}) (cons 71 1) (cons 72 1) '
            f'(cons 1 "{literal(value)}"))) (setq cap_added (1+ cap_added)))'
        )
        manifest.append({"crop": crop, "line": line, "bom_row": row,
                         "text": value, "outline_count": len(handles),
                         "pdf_bbox": [left, glyph_top, right, max(b[3] for b in glyph_boxes)]})
    commands += [
        '(princ (strcat "\\nCAP_ADDED=" (itoa cap_added) " CAP_MOVED=" (itoa cap_moved)))',
        '_.QSAVE',
        '_.QUIT',
        '_N',
    ]
    script = OUTPUT.with_suffix(".scr")
    script.write_text("\n".join(commands) + "\n", encoding="ascii")
    log = OUTPUT.with_suffix(".log")
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode("utf-16-le", errors="replace")
    expected = f"CAP_ADDED={len(DECISIONS)} CAP_MOVED={len(used_handles)}"
    if expected not in report:
        raise RuntimeError(f"DWG conversion incomplete; expected {expected} in {log}")
    OUTPUT.with_suffix(".json").write_text(
        json.dumps({"source": str(SOURCE), "converted": manifest,
                    "policy": "BOM-backed lines only; unverified caption lines retained as outlines"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(expected)


if __name__ == "__main__":
    build()
