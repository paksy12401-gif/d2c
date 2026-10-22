"""Fill the three remaining symbol captions using the restored circuit list."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from blockify_pdf_symbols import CORE_CONSOLE, audit, run_console
from bom_ocr_to_dwg import literal

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "verified_dwg/20261013"
SOURCE = FOLDER / "symbols_59_with_bom_captions_and_circuit_table.dwg"
OUTPUT = FOLDER / "symbols_59_with_circuit_table_all_captions.dwg"
TABLE_CSV = SOURCE.with_suffix(".csv")
PDF_HEIGHT = 2383.93994140625

# Rectangles contain only the still-visible black glyph polylines. Each first
# line is independently present in the circuit table; part/rear-cover IDs are
# cross-checked against the BOM OCR table.
GROUPS = [
    {"name": "left_lower", "bounds": (150, 1650, 480, 1685),
     "x": 160.5, "tops": [1651.8, 1662.4, 1673.0],
     "texts": ["부품명 : 13543056_X604X604_E603_LT_X2_X2",
               "CONNECTOR NO : 13543056", "FSG CLIP NO (28번) : 13352244"],
     "bom_ids": ["13543056", "13352244"]},
    {"name": "right_lower", "bounds": (4940, 1790, 5220, 1813),
     "x": 4949.3, "tops": [1791.7, 1802.4],
     "texts": ["부품명 : 13590425_Y572LEFT_FRONT_X1",
               "CONNECTOR NO : 13590425"],
     "bom_ids": ["13590425"]},
    {"name": "upper_middle", "bounds": (3100, 1312, 3350, 1345),
     "x": 3107.6, "tops": [1313.2, 1323.8, 1334.5],
     "texts": ["부품명 : 13512366_M651LEFT_FRONT_X1",
               "CONNECTOR NO : 13512366", "REAR COVER NO : OPU88567"],
     "bom_ids": ["13512366", "OPU88567"]},
]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not SOURCE.is_file() or not CORE_CONSOLE.is_file():
        raise FileNotFoundError("Source DWG or AutoCAD Core Console missing")
    with TABLE_CSV.open(encoding="utf-8-sig", newline="") as stream:
        circuit_rows = list(csv.DictReader(stream))
    names = {row[column] for row in circuit_rows
             for column in ("FROM_CONNECTOR", "TO_CONNECTOR")}
    bom = json.loads((ROOT / "blockified_dwg/bom_cell_ocr.json").read_text(encoding="utf-8"))
    bom_ids = {cell["text"].strip() for cell in bom["cells"] if cell["column"] == 3}
    for group in GROUPS:
        part = group["texts"][0].split(" : ", 1)[1]
        if part not in names or not set(group["bom_ids"]) <= bom_ids:
            raise AssertionError(f"Caption missing from circuit table or BOM: {group['name']}")

    shutil.copy2(SOURCE, OUTPUT)
    audit_csv = audit(OUTPUT, FOLDER, "remaining_captions_before")
    entities = list(csv.DictReader(audit_csv.open(encoding="utf-8", newline="")))
    commands = ["(setq extra_moved 0 extra_added 0)"]
    manifest = []
    used = set()
    for group in GROUPS:
        bx0, by0, bx1, by1 = group["bounds"]
        handles = []
        for entity in entities:
            if entity["type"] != "LWPOLYLINE" or entity["layer"] != "PDF_Geomertry" or entity["color"] != "7":
                continue
            try:
                a, b, c, d = (float(entity[key]) for key in
                              ("minx", "miny", "maxx", "maxy"))
            except ValueError:
                continue
            x0, y0, x1, y1 = a * 72, PDF_HEIGHT - d * 72, c * 72, PDF_HEIGHT - b * 72
            if bx0 <= x0 <= x1 <= bx1 and by0 <= y0 <= y1 <= by1:
                handle = entity["handle"]
                if handle in used:
                    raise AssertionError(f"Outline belongs to two captions: {handle}")
                handles.append(handle)
                used.add(handle)
        if len(handles) < 30:
            raise AssertionError(f"Caption outlines incomplete: {group['name']}: {len(handles)}")
        for handle in handles:
            commands.append(
                f'(if (setq cap_ent (handent "{handle}")) '
                f'(progn (setq cap_data (entget cap_ent)) '
                f'(if (entmod (subst (cons 8 "CAPTION_ORIGINAL_OUTLINES") '
                f'(assoc 8 cap_data) cap_data)) (setq extra_moved (1+ extra_moved)))))'
            )
        x = group["x"] / 72
        width = (bx1 - bx0 + 15) / 72
        for top, value in zip(group["tops"], group["texts"]):
            y = (PDF_HEIGHT - top) / 72
            commands.append(
                f'(if (entmakex (list (cons 0 "MTEXT") (cons 100 "AcDbEntity") '
                f'(cons 8 "CAPTION_LLM_TEXT") (cons 100 "AcDbMText") '
                f'(cons 10 (list {x:.8f} {y:.8f} 0.0)) (cons 40 0.130000) '
                f'(cons 41 {width:.8f}) (cons 71 1) (cons 72 1) '
                f'(cons 1 "{literal(value)}"))) (setq extra_added (1+ extra_added)))'
            )
        manifest.append({**group, "hidden_outline_count": len(handles)})
    added = sum(len(group["texts"]) for group in GROUPS)
    commands += [
        '_.-LAYER', '_OFF', 'CAPTION_ORIGINAL_OUTLINES', '',
        '(princ (strcat "\\nEXTRA_ADDED=" (itoa extra_added) '
        '" EXTRA_MOVED=" (itoa extra_moved)))',
        '_.QSAVE', '_.QUIT', '_N',
    ]
    script = OUTPUT.with_suffix(".scr")
    script.write_text("\n".join(commands) + "\n", encoding="ascii")
    log = OUTPUT.with_suffix(".log")
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode("utf-16-le", errors="replace")
    expected = f"EXTRA_ADDED={added} EXTRA_MOVED={len(used)}"
    if expected not in report:
        raise RuntimeError(f"AutoCAD did not save all caption changes: {log}")
    OUTPUT.with_suffix(".json").write_text(
        json.dumps({"source": str(SOURCE), "added": manifest,
                    "reason": "Three unconverted PDF captions matched circuit table names"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("DWG", OUTPUT)
    print(expected)


if __name__ == "__main__":
    main()
