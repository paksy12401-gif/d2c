"""Put cell-by-cell BOM OCR into a copy of the checked symbol DWG.

Run after tmp/bom_recognize.py has produced blockified_dwg/bom_cell_ocr.json.
Original outline glyphs are kept on an off layer so the conversion is reversible.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import shutil

from blockify_pdf_symbols import CORE_CONSOLE, audit, run_console


ROOT = Path(__file__).resolve().parent
SOURCE_DWG = ROOT / "verified_dwg/20261013/symbols_59_pipe_separate_exact_geometry.dwg"
OCR_JSON = ROOT / "blockified_dwg/bom_cell_ocr.json"
OUTPUT = ROOT / "verified_dwg/20261013/symbols_59_with_bom_ocr_v3.dwg"
MANIFEST = OUTPUT.with_suffix(".csv")
MIN_CONFIDENCE = 0.70
PDF_HEIGHT = 2383.93994140625
CROP_X, CROP_Y, RENDER_SCALE = 5000.0, 225.0, 2.0
OUTLINE_LAYER = "BOM_ORIGINAL_OUTLINES"
TEXT_LAYER = "BOM_OCR_TEXT"

# Checked against enlarged cells of this sample PDF.  Keep uncertain raw
# metadata fields unchanged rather than guessing at their separators.
VERIFIED_TEXT = {
    (30, 4): "BKC80763",
    (26, 4): "ASM CONN 8 F OCS 1.5",
    (25, 2): "DELPHI - 35016344 - CONNECTOR ASM - UNSEALED",
    (25, 4): "ASM CONN 16 F 0.64 CPA OCS",
    (23, 2): "DELPHI-13610095-CONNECTOR ASM-WRG HARN",
    (23, 4): "ASM CONN 2 F KAIZEN 0.64 GRN SLD",
    (22, 2): "SUMITOMO 6099-0288 CONNECTOR ASM-UNSEALED",
    (20, 2): "TYCO-1-2296695-1 CONNECTOR ASM-INLINE",
    (20, 4): "CONNECTOR ASM-INLINE",
    (19, 2): "YAZAKI-7297-2973-30-CONNECTOR ASM-INLINE",
    (19, 4): "BDR13636",
    (18, 2): "YAZAKI-7297-6353-30-CONNECTOR ASM-INLINE",
    (18, 4): "BOE53042",
    (17, 1): "ADN03474.GEO_FIN001.001999",
    (17, 2): "YAZAKI 7075-1839-30 GROMMET WIPER",
    (17, 3): "ADN03474",
    (17, 4): "WIPER GROMMET 7075-1839-30",
    (16, 4): "BSB04473",
    (15, 2): "YAZAKI-7283-8397-40-CONNECTOR ASM-INLINE",
    (15, 4): "13592509",
    (14, 2): "HELLERMANNTYTON-157-00154-CLIP-WRG HARN",
    (14, 4): "HELLERMANNTYTON-157-00154-CLIP-WRG HARN HT# 157-00154 HT Description 11-0345-001-CSU",
    (13, 2): "HELLERMANNTYTON-156-02467-CLIP-WRG HARN",
    (13, 4): "HELLERMANNTYTON-156-02467-CLIP-WRG HARN",
    (12, 2): "ARAYMOND-230347001-CLIP-WRG HARN",
    (11, 4): "HellermannTyton SPN 157-00128",
    (10, 2): "HELLERMANNTYTON-157-00224-CLIP-WRG HARN",
    (8, 4): "MX150 1X2 F CS POLE BLK w/ CPA 5.0CL",
    (7, 2): "STRAP ASM-WRG HARN",
    (2, 4): "BJY07155",
    (1, 2): "AVERY-DENNISON-08401-0-CLIP-WRG HARN",
    (1, 4): "AVERY-DENNISON-08401-0-CLIP-WRG HARN",
}


def normalized_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper()).replace("O", "0").replace("I", "1")


def canonical_part_name(raw: str, db_id: str) -> str:
    parts = raw.upper().replace(" ", "").split(".", 1)
    if len(parts) != 2 or normalized_id(parts[0]) != normalized_id(db_id):
        return ""
    suffix = parts[1].replace("GE0", "GEO").replace("F1N", "FIN")
    match = re.fullmatch(r"GEO_(FIN|ENV)001\.([0-9O]{6})", suffix)
    if not match:
        return ""
    return f"{db_id}.GEO_{match[1]}001.{match[2].replace('O', '0')}"


def literal(value: str) -> str:
    value = " ".join(value.strip().split())
    value = value.translate(str.maketrans({"–": "-", "—": "-", "−": "-", "×": "X"}))
    # AutoCAD script is ASCII; retain all other Unicode as MTEXT escapes.
    value = "".join(char if ord(char) < 128 else f"\\U+{ord(char):04X}" for char in value)
    return value.replace("\\", "\\\\").replace('"', '\\"')


def cell_bounds(cell: dict) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = cell["bbox_px"]
    return ((CROP_X + x0 / RENDER_SCALE) / 72,
            (PDF_HEIGHT - CROP_Y - y1 / RENDER_SCALE) / 72,
            (CROP_X + x1 / RENDER_SCALE) / 72,
            (PDF_HEIGHT - CROP_Y - y0 / RENDER_SCALE) / 72)


def accepted_cells(cells: list[dict]) -> list[dict]:
    by_row = {(cell["row"], cell["column"]): cell for cell in cells}
    assert len(by_row) == 160
    result = []
    headers = ["PC NO", "PART NAME", "DB PART NAME", "DB PART NO", "DB PART DESC"]
    for row in [*range(31, 0, -1), "header"]:
        part = by_row[row, 1]["text"].split(".", 1)[0].strip()
        db_id = by_row[row, 3]["text"].strip()
        ids_match = row == "header" or (normalized_id(part) == normalized_id(db_id)
                                        and len(normalized_id(db_id)) == 8)
        for col in range(5):
            cell = dict(by_row[row, col])
            text = " ".join(cell["text"].split())
            score = cell["confidence"]
            keep = bool(text) and score >= MIN_CONFIDENCE and row != 27
            if col == 0 and row != "header":
                keep &= text == str(row)
                text = str(row)
            if col in (1, 3) and row != "header":
                keep &= ids_match
            if col == 1 and row != "header":
                text = canonical_part_name(text, db_id)
                keep &= bool(text)
            if row == "header":
                text = headers[col]
            elif keep:
                text = VERIFIED_TEXT.get((row, col), text)
            cell["accepted"] = bool(keep)
            cell["text_to_insert"] = text if keep else ""
            result.append(cell)
    assert [c["row"] for c in result if c["column"] == 0][:31] == list(range(31, 0, -1))
    return result


def original_glyph_handles(audit_csv: Path, cells: list[dict]) -> dict[tuple, list[str]]:
    groups = {(c["row"], c["column"]): [] for c in cells if c["accepted"]}
    boxes = {key: cell_bounds(next(c for c in cells if (c["row"], c["column"]) == key))
             for key in groups}
    with audit_csv.open(encoding="utf-8", newline="") as stream:
        for entity in csv.DictReader(stream):
            if (entity["type"] != "LWPOLYLINE" or entity["layer"] not in
                    {"PDF_Geomertry", "PDF_Geometry"} or entity["color"] not in {"18", "256"}):
                continue
            try:
                ex0, ey0, ex1, ey1 = (float(entity[k]) for k in ("minx", "miny", "maxx", "maxy"))
            except ValueError:
                continue
            for key, (x0, y0, x1, y1) in boxes.items():
                if x0 <= ex0 <= ex1 <= x1 and y0 <= ey0 <= ey1 <= y1:
                    groups[key].append(entity["handle"])
                    break
    handles = [handle for group in groups.values() for handle in group]
    assert len(handles) == len(set(handles)), "One outline was assigned to multiple OCR cells"
    return groups


def script_for(cells: list[dict], groups: dict[tuple, list[str]]) -> str:
    commands = [
        '(entmake (list (cons 0 "LAYER") (cons 2 "BOM_ORIGINAL_OUTLINES") '
        '(cons 70 0) (cons 62 -8) (cons 6 "Continuous")))',
        '(entmake (list (cons 0 "LAYER") (cons 2 "BOM_OCR_TEXT") '
        '(cons 70 0) (cons 62 7) (cons 6 "Continuous")))',
        '(setq bom_moved 0 bom_added 0)',
    ]
    for cell in cells:
        if not cell["accepted"]:
            continue
        for handle in groups[cell["row"], cell["column"]]:
            commands.append(
                f'(if (setq bom_ent (handent "{handle}")) '
                f'(progn (setq bom_data (entget bom_ent)) '
                f'(if (entmod (subst (cons 8 "{OUTLINE_LAYER}") '
                f'(assoc 8 bom_data) bom_data)) (setq bom_moved (1+ bom_moved)))))')
        x0, y0, x1, y1 = cell_bounds(cell)
        # MTEXT insertion is top-left. The small inset keeps text off the grid.
        x = x0 + 0.035
        y = y1 - 0.045
        width = x1 - x0 - 0.07
        height = 0.18
        value = literal(cell["text_to_insert"])
        commands.append(
            f'(if (entmakex (list (cons 0 "MTEXT") (cons 100 "AcDbEntity") '
            f'(cons 8 "{TEXT_LAYER}") (cons 100 "AcDbMText") '
            f'(cons 10 (list {x:.8f} {y:.8f} 0.0)) (cons 40 {height:.6f}) '
            f'(cons 41 {width:.8f}) (cons 71 1) (cons 72 1) (cons 1 "{value}"))) '
            f'(setq bom_added (1+ bom_added)))')
    commands += [
        '_.-LAYER', '_OFF', OUTLINE_LAYER, '',
        '(princ (strcat "\\nBOM_OCR_ADDED=" (itoa bom_added) '
        '" BOM_OUTLINES_MOVED=" (itoa bom_moved)))',
        '_.QSAVE',
        '_.QUIT',
        '_N',
    ]
    return "\n".join(commands) + "\n"


def main() -> None:
    if not all(p.is_file() for p in (SOURCE_DWG, OCR_JSON, CORE_CONSOLE)):
        raise FileNotFoundError("Source DWG, OCR JSON, or AutoCAD Core Console is missing")
    cells = accepted_cells(json.loads(OCR_JSON.read_text(encoding="utf-8"))["cells"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    shutil.copy2(SOURCE_DWG, OUTPUT)
    (OUTPUT.parent / "entities_bom_before.csv").unlink(missing_ok=True)
    audit_csv = audit(OUTPUT, OUTPUT.parent, "bom_before")
    groups = original_glyph_handles(audit_csv, cells)
    for cell in cells:
        cell["hidden_outline_count"] = len(groups.get((cell["row"], cell["column"]), []))
        cell["bounds_dwg"] = cell_bounds(cell)
    script = OUTPUT.with_suffix(".scr")
    script.write_text(script_for(cells, groups), encoding="ascii")
    log = OUTPUT.with_suffix(".log")
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode("utf-16-le", errors="replace")
    expected_added = sum(c["accepted"] for c in cells)
    expected_moved = sum(map(len, groups.values()))
    if f"BOM_OCR_ADDED={expected_added} BOM_OUTLINES_MOVED={expected_moved}" not in report:
        raise RuntimeError(f"AutoCAD did not create every MTEXT/outline move; inspect {log}")
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row", "column", "text", "confidence",
                                                   "accepted", "text_to_insert", "hidden_outline_count",
                                                   "bbox_px", "bounds_dwg"])
        writer.writeheader()
        writer.writerows(cells)
    print("DWG:", OUTPUT)
    print("OCR cells accepted:", sum(c["accepted"] for c in cells), "/ 160")
    print("Outlines moved:", sum(map(len, groups.values())))
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
