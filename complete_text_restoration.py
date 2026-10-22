"""Finish the sample drawing's editable descriptive text.

Only the listed, visually checked PDF text regions are replaced.  The source
outlines stay in the DWG on an explicitly switched-off layer, so the operation
is reversible and cannot remove adjacent leaders, pipes, or symbols.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from uuid import uuid4

from blockify_pdf_symbols import CORE_CONSOLE, audit, run_console
from bom_ocr_to_dwg import literal


ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "verified_dwg/20261014"
SOURCE = FOLDER / "symbols_59_with_editable_work_notes.dwg"
OUTPUT = FOLDER / "symbols_59_text_corrected.dwg"
PDF_HEIGHT = 2383.93994140625
OUTLINE_LAYER = "ADDITIONAL_TEXT_ORIGINAL_OUTLINES"
TEXT_LAYER = "ADDITIONAL_EDITABLE_TEXT"

# Coordinates are PDF points.  These rectangles contain text glyphs only;
# the one leader crossing the middle note is excluded by its width.
# Each tuple is (PDF x, PDF top, value, MTEXT height).
REGIONS = [
    {"name": "caption_04", "box": (2047, 956, 2378, 1002), "count": 82,
     "lines": [(2059.7, 965.8, "부자재 리스트 17번", .16),
               (2059.7, 980.4, "GROMMET 적용후 2번 COLDMELT 적용할것", .16)]},
    {"name": "caption_11", "box": (1976, 1495, 2183, 1527), "count": 31,
     "lines": [(1988.9, 1504.4, "FLEECE TAPE 2회 적용", .16)]},
    {"name": "caption_12", "box": (3110, 1552, 3277, 1581), "count": 31,
     "lines": [(3122.7, 1561.1, "FLEECE TAPE 2회 적용", .13)]},
    {"name": "caption_13", "box": (1990, 1807, 2160, 1852), "count": 52,
     "lines": [(2003.0, 1816.2, "부자재 리스트 2번", .16),
               (2003.0, 1831.7, "COLDMELT 적용할것", .16)]},
    {"name": "upper_callout", "box": (3430, 400, 3590, 425), "count": 54,
     "lines": [(3430.3, 402.4, "SLEEVE 적용할것.", .13),
               (3430.3, 413.8, "P/NO : SA-08D-0BD-BK", .13)]},
    {"name": "middle_callout", "box": (3370, 1330, 3710, 1400),
     "count": 89, "max_glyph_width": 35,
     "lines": [(3409.9, 1334.3, "FLEECE TAPE 2회 적용", .16),
               (3414.7, 1362.7, "부자재 리스트 2번 : COLDMELT 적용할것", .16)]},
    {"name": "left_note", "box": (1940, 1620, 2140, 1665),
     "count": 31, "max_bottom": 1645.5,
     "lines": [(1950.6, 1632.3, "FLEECE TAPE 2회 적용", .16)]},
    {"name": "splice_point", "box": (2690, 1590, 2810, 1620),
     "count": 14, "color": "7",
     "lines": [(2702.1, 1597.0, "SPLICE POINT", .13)]},
    {"name": "bom_title", "box": (5250, 180, 6980, 270), "count": 65,
     "lines": [(5262.9, 186.3,
                "부자재 LIST ( 커넥터, B/CABLE, 그로멧, TAPE등 )", .85)]},
]

OLD_OUTLINE_LAYERS = (
    "CAPTION_ORIGINAL_OUTLINES",
    "BOM_ORIGINAL_OUTLINES",
    "WORK_NOTE_ORIGINAL_OUTLINES",
)


def pdf_bounds(row: dict[str, str]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = (float(row[key]) for key in
                              ("minx", "miny", "maxx", "maxy"))
    return minx * 72, PDF_HEIGHT - maxy * 72, maxx * 72, PDF_HEIGHT - miny * 72


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not SOURCE.is_file() or not CORE_CONSOLE.is_file():
        raise FileNotFoundError("Source DWG or AutoCAD Core Console is missing")
    shutil.copy2(SOURCE, OUTPUT)
    before = audit(OUTPUT, FOLDER, f"additional_text_before_{uuid4().hex[:8]}")
    with before.open(encoding="utf-8", newline="") as stream:
        entities = list(csv.DictReader(stream))

    commands = [
        f'(entmake (list (cons 0 "LAYER") (cons 2 "{OUTLINE_LAYER}") '
        '(cons 70 0) (cons 62 7) (cons 6 "Continuous")))',
        f'(entmake (list (cons 0 "LAYER") (cons 2 "{TEXT_LAYER}") '
        '(cons 70 0) (cons 62 7) (cons 6 "Continuous")))',
        '(setq additional_moved 0 additional_added 0)',
    ]
    used: set[str] = set()
    manifest = []
    for region in REGIONS:
        left, top, right, bottom = region["box"]
        glyphs = []
        for entity in entities:
            if (entity["type"] != "LWPOLYLINE"
                    or entity["layer"] != "PDF_Geomertry"
                    or entity["color"] != region.get("color", "7")):
                continue
            try:
                x0, y0, x1, y1 = pdf_bounds(entity)
            except ValueError:
                continue
            if not (left <= x0 <= x1 <= right and top <= y0 <= y1 <= bottom):
                continue
            if x1 - x0 > region.get("max_glyph_width", float("inf")):
                continue
            if y1 > region.get("max_bottom", float("inf")):
                continue
            glyphs.append(entity["handle"])
        if len(glyphs) != region["count"]:
            raise AssertionError(f'{region["name"]}: expected {region["count"]} '
                                 f'outlines, got {len(glyphs)}')
        for handle in glyphs:
            if handle in used:
                raise AssertionError(f"Outline selected twice: {handle}")
            used.add(handle)
            commands.append(
                f'(if (setq text_ent (handent "{handle}")) '
                '(progn (setq text_data (entget text_ent)) '
                f'(if (entmod (subst (cons 8 "{OUTLINE_LAYER}") '
                '(assoc 8 text_data) text_data)) '
                '(setq additional_moved (1+ additional_moved)))))'
            )
        for x, y, value, height in region["lines"]:
            dwg_x, dwg_y = x / 72, (PDF_HEIGHT - y) / 72
            width = (right - x + 80) / 72
            commands.append(
                '(if (entmakex (list (cons 0 "MTEXT") (cons 100 "AcDbEntity") '
                f'(cons 8 "{TEXT_LAYER}") (cons 100 "AcDbMText") '
                f'(cons 10 (list {dwg_x:.8f} {dwg_y:.8f} 0.0)) '
                f'(cons 40 {height:.6f}) (cons 41 {width:.8f}) '
                '(cons 71 1) (cons 72 1) '
                f'(cons 1 "{literal(value)}"))) '
                '(setq additional_added (1+ additional_added)))'
            )
        manifest.append({"name": region["name"], "pdf_box": region["box"],
                         "outlines_hidden": len(glyphs),
                         "texts": [line[2] for line in region["lines"]]})

    # An entmake colour of -8 did not survive DWG save as an OFF layer.
    # Explicit -LAYER commands do, and are checked after reopening below.
    for layer in (*OLD_OUTLINE_LAYERS, OUTLINE_LAYER):
        commands += ['_.-LAYER', '_OFF', layer, '']
    commands += [
        '(princ (strcat "\\nADDITIONAL_ADDED=" (itoa additional_added) '
        '" ADDITIONAL_MOVED=" (itoa additional_moved)))',
        '_.QSAVE', '_.QUIT', '_N',
    ]
    script = OUTPUT.with_suffix(".scr")
    script.write_text("\n".join(commands) + "\n", encoding="ascii")
    log = OUTPUT.with_suffix(".log")
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode("utf-16-le", errors="replace")
    added = sum(len(region["lines"]) for region in REGIONS)
    expected = f"ADDITIONAL_ADDED={added} ADDITIONAL_MOVED={len(used)}"
    if expected not in report:
        raise RuntimeError(f"AutoCAD did not complete the change: {log}")

    after = audit(OUTPUT, FOLDER, f"additional_text_after_{uuid4().hex[:8]}")
    with after.open(encoding="utf-8", newline="") as stream:
        after_rows = list(csv.DictReader(stream))
    if sum(row["layer"] == TEXT_LAYER and row["type"] == "MTEXT"
           for row in after_rows) != added:
        raise AssertionError("Saved DWG is missing additional MTEXT")
    if sum(row["layer"] == OUTLINE_LAYER and row["type"] == "LWPOLYLINE"
           for row in after_rows) != len(used):
        raise AssertionError("Saved DWG did not preserve the original glyphs")
    if sum(row["type"] == "INSERT" for row in after_rows) != 59:
        raise AssertionError("One or more symbol blocks changed")

    verify_script = OUTPUT.with_name("verify_additional_layers.scr")
    verify_script.write_text("\n".join(
        f'(princ (strcat "\\nLAYER_{layer}=" '
        f'(itoa (cdr (assoc 62 (tblsearch "LAYER" "{layer}"))))))'
        for layer in (*OLD_OUTLINE_LAYERS, OUTLINE_LAYER)
    ) + '\n_.QUIT\n_N\n', encoding="ascii")
    verify_log = OUTPUT.with_name("verify_additional_layers.log")
    run_console(OUTPUT, verify_script, verify_log)
    verify_report = verify_log.read_bytes().decode("utf-16-le", errors="replace")
    for layer in (*OLD_OUTLINE_LAYERS, OUTLINE_LAYER):
        if f"LAYER_{layer}=-7" not in verify_report:
            raise AssertionError(f"Original outlines remain visible: {layer}")

    OUTPUT.with_suffix(".json").write_text(json.dumps(
        {"source": str(SOURCE), "regions": manifest,
         "additional_editable_lines": added, "hidden_outlines": len(used),
         "original_layers_off": list((*OLD_OUTLINE_LAYERS, OUTLINE_LAYER)),
         "scope": "descriptive notes, captions, callouts, and BOM title",
         "excluded": "dimensions, circled numbers, and struck-out BOM row 27"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(expected)


if __name__ == "__main__":
    main()
