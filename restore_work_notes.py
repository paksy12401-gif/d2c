"""Replace the sample drawing's outlined work notes with editable MTEXT.

The source PDF is retained as the visual reference.  This script targets only
the five work-note lines at their measured PDF coordinates; it does not try to
interpret arbitrary polylines in other drawings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from blockify_pdf_symbols import CORE_CONSOLE, audit, run_console
from bom_ocr_to_dwg import literal


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "source_inputs" / "DWG 객체 변환 샘플.pdf"
SOURCE = ROOT / "verified_dwg/20261013/symbols_59_with_circuit_table_all_captions.dwg"
OUTPUT = ROOT / "verified_dwg/20261014/symbols_59_with_editable_work_notes.dwg"
PDF_HEIGHT = 2383.93994140625
LEFT, RIGHT = 6060.0, 7140.0
OUTLINE_LAYER = "WORK_NOTE_ORIGINAL_OUTLINES"
TEXT_LAYER = "WORK_NOTE_MTEXT"

# Bounds and text were read from a 2x PDF render.  The OCR result is retained
# separately as evidence, but it missed parts of these outlined characters.
# (name, PDF top, PDF bottom, expected polyline count, exact transcription)
NOTES = [
    ("heading", 1300, 1350, 15, "작업 주의"),
    ("item_1", 1360, 1415, 90,
     "1. 모든 분기부위는 크로스 테핑 진행할것 ( FLEECE TAPE 적용 )"),
    ("item_2", 1420, 1470, 93,
     "2. 모든 B/CABLE 구간 작업시 FLEECE TAPE 2회 적용후 작업할것."),
    ("item_3", 1480, 1530, 62,
     "3. 부자재 LIST 참조하여 적용할것."),
    ("item_4", 1540, 1590, 83,
     "4. 별도 명시되지 않은 외장재는 FLEECE ALL TAP'G 할것"),
]


def _pdf_box(entity: dict[str, str]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(entity[key]) for key in
                      ("minx", "miny", "maxx", "maxy"))
    return x0 * 72, PDF_HEIGHT - y1 * 72, x1 * 72, PDF_HEIGHT - y0 * 72


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path in (PDF, SOURCE, CORE_CONSOLE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, OUTPUT)
    before = audit(OUTPUT, OUTPUT.parent, "work_note_before")
    with before.open(encoding="utf-8", newline="") as stream:
        entities = list(csv.DictReader(stream))

    commands = [
        f'(entmake (list (cons 0 "LAYER") (cons 2 "{OUTLINE_LAYER}") '
        '(cons 70 0) (cons 62 -8) (cons 6 "Continuous")))',
        f'(entmake (list (cons 0 "LAYER") (cons 2 "{TEXT_LAYER}") '
        '(cons 70 0) (cons 62 7) (cons 6 "Continuous")))',
        '(setq note_moved 0 note_added 0)',
    ]
    used: set[str] = set()
    manifest = []
    for name, top, bottom, expected_count, value in NOTES:
        glyphs = []
        for entity in entities:
            if (entity["type"] != "LWPOLYLINE"
                    or entity["layer"] != "PDF_Geomertry"
                    or entity["color"] != "7"):
                continue
            try:
                x0, y0, x1, y1 = _pdf_box(entity)
            except ValueError:
                continue
            if LEFT <= x0 <= x1 <= RIGHT and top <= y0 <= y1 <= bottom:
                glyphs.append((entity["handle"], x0, y0, x1, y1))
        if len(glyphs) != expected_count:
            raise AssertionError(
                f"{name}: expected {expected_count} outlines, got {len(glyphs)}"
            )
        for handle, *_ in glyphs:
            if handle in used:
                raise AssertionError(f"Outline selected twice: {handle}")
            used.add(handle)
            commands.append(
                f'(if (setq note_ent (handent "{handle}")) '
                f'(progn (setq note_data (entget note_ent)) '
                f'(if (entmod (subst (cons 8 "{OUTLINE_LAYER}") '
                '(assoc 8 note_data) note_data)) '
                '(setq note_moved (1+ note_moved)))))'
            )
        left = min(g[1] for g in glyphs)
        glyph_top = min(g[2] for g in glyphs)
        right = max(g[3] for g in glyphs)
        glyph_bottom = max(g[4] for g in glyphs)
        x, y = left / 72, (PDF_HEIGHT - glyph_top) / 72
        width = (right - left + 120) / 72
        height = 0.36
        commands.append(
            '(if (entmakex (list (cons 0 "MTEXT") (cons 100 "AcDbEntity") '
            f'(cons 8 "{TEXT_LAYER}") (cons 100 "AcDbMText") '
            f'(cons 10 (list {x:.8f} {y:.8f} 0.0)) '
            f'(cons 40 {height:.6f}) (cons 41 {width:.8f}) '
            '(cons 71 1) (cons 72 1) '
            f'(cons 1 "{literal(value)}"))) '
            '(setq note_added (1+ note_added)))'
        )
        manifest.append({"name": name, "text": value,
                         "outline_count": len(glyphs),
                         "pdf_bbox": [left, glyph_top, right, glyph_bottom],
                         "dwg_insert": [x, y], "text_height": height})

    commands.extend([
        '_.-LAYER', '_OFF', OUTLINE_LAYER, '',
        '(princ (strcat "\\nNOTE_ADDED=" (itoa note_added) '
        '" NOTE_MOVED=" (itoa note_moved)))',
        '_.QSAVE', '_.QUIT', '_N',
    ])
    script = OUTPUT.with_suffix(".scr")
    script.write_text("\n".join(commands) + "\n", encoding="ascii")
    log = OUTPUT.with_suffix(".log")
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode("utf-16-le", errors="replace")
    expected = f"NOTE_ADDED={len(NOTES)} NOTE_MOVED={len(used)}"
    if expected not in report:
        raise RuntimeError(f"DWG conversion incomplete; expected {expected}: {log}")

    after = audit(OUTPUT, OUTPUT.parent, "work_note_after")
    with after.open(encoding="utf-8", newline="") as stream:
        after_rows = list(csv.DictReader(stream))
    if sum(row["layer"] == TEXT_LAYER and row["type"] == "MTEXT"
           for row in after_rows) != len(NOTES):
        raise AssertionError("The saved DWG does not contain all five MTEXT lines")
    if sum(row["layer"] == OUTLINE_LAYER and row["type"] == "LWPOLYLINE"
           for row in after_rows) != len(used):
        raise AssertionError("The saved DWG did not preserve all outline polylines")
    OUTPUT.with_suffix(".json").write_text(
        json.dumps({"source": str(SOURCE), "source_pdf": str(PDF),
                    "transcription": "visual PDF reading with local OCR cross-check",
                    "notes": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT)
    print(expected)


if __name__ == "__main__":
    main()
