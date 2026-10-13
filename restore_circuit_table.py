"""Rebuild the 71-row raster circuit table as editable AutoCAD geometry.

The source PDF contains one table image. Its pixel-identical repeated cells
provide a stronger check than raw OCR for connector names. All 720 table cells
are mapped back to the original image coordinates in a copy of the DWG.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from blockify_pdf_symbols import CORE_CONSOLE, run_console
from bom_ocr_to_dwg import literal

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT / "blockified_dwg/right_circuit_source.png"
OCR = ROOT / "blockified_dwg/right_circuit_ocr_raw.json"
SOURCE = ROOT / "verified_dwg/20261013/symbols_59_with_bom_and_captions_llm_v4.dwg"
OUTPUT = ROOT / "verified_dwg/20261013/symbols_59_with_bom_captions_and_circuit_table.dwg"
CSV = OUTPUT.with_suffix(".csv")
REPORT = OUTPUT.with_suffix(".json")
XS = [0, 72, 140, 186, 251, 346, 628, 692, 974, 1070, 1204]
HEADERS = ["NO", "WIRE", "SQUA", "COL", "MAT", "FROM_CONNECTOR",
           "PIN ID1", "TO_CONNECTOR", "PIN ID2", "TWIST & SHIELD"]
PDF_HEIGHT = 2383.93994140625
IMAGE_BBOX = (8011.72998046875, 95.8701171875,
              11149.6796875, 2208.27001953125)

# Representative cells read from the embedded image. Every repeated cell is
# required to have the exact same pixel hash before receiving this value.
CONNECTOR_REPRESENTATIVES = {
    (5, 1): "13512366_M651LEFT_FRONT_X1",
    (5, 2): "13551224_X11X500_FL_X1",
    (5, 3): "13535680_X644X505_FL_X1",
    (5, 7): "13519739_Y626LEFT_FRONT_MAO_X1",
    (5, 11): "13590425_Y572LEFT_FRONT_X1",
    (5, 12): "13532124_X604X604_E603_LT_X1_X1",
    (5, 18): "13543056_X604X604_E603_LT_X2_X2",
    (5, 19): "13514590_X604X604_A471_X1",
    (5, 36): "13592509_A470LEFT_FRONT_X1",
    (5, 41): "13591047_Y626LEFT_FRONT_MAO_X2",
    (7, 5): "19151432_B63DOOR_LFR_X1",
    (7, 6): "STD_X117SP3840_X1",
    (7, 14): "13590001_M617LEFT_FRONT_X1",
    (7, 24): "STD_X117SP2150W_X1",
    (7, 26): "13582917_B198LEFT_FRONT_X1",
    (7, 36): "STD_X117SP2150D_X1",
}
TWIST_REPRESENTATIVES = {
    1: "0", 5: "TW2132_W61", 17: "TW1874_W61",
    18: "SH4731_1", 26: "TW118_W615",
}
COLOR_CORRECTIONS = {13: "WH/VT", 50: "BU/VT"}


def image_cell(gray: np.ndarray, row: int, col: int) -> np.ndarray:
    return gray[row * 22 + 2:(row + 1) * 22 - 1, XS[col] + 2:XS[col + 1] - 1]


def digest(cell: np.ndarray) -> str:
    return hashlib.sha256(cell.tobytes()).hexdigest()


def pixel_to_dwg(x: float, y: float) -> tuple[float, float]:
    px0, py0, px1, py1 = IMAGE_BBOX
    pdf_x = px0 + x * (px1 - px0) / 1204
    pdf_y = py0 + y * (py1 - py0) / 1584
    return pdf_x / 72, (PDF_HEIGHT - pdf_y) / 72


def resolved_cells() -> list[dict]:
    gray = cv2.imdecode(np.fromfile(IMAGE, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray.shape != (1584, 1204):
        raise ValueError(f"Unexpected table image size: {gray.shape}")
    raw = json.loads(OCR.read_text(encoding="utf-8"))["cells"]
    if len(raw) != 720:
        raise ValueError("Expected OCR of 72 rows x 10 columns")

    connectors = {}
    for (col, row), text in CONNECTOR_REPRESENTATIVES.items():
        key = digest(image_cell(gray, row, col))
        if key in connectors and connectors[key] != text:
            raise AssertionError("One pixel-identical connector has two names")
        connectors[key] = text
    twists = {digest(image_cell(gray, row, 9)): text
              for row, text in TWIST_REPRESENTATIVES.items()}
    if len(connectors) != 16 or len(twists) != 5:
        raise AssertionError("Representative cells are not unique")

    result = []
    for item in raw:
        row, col = item["row"], item["column"]
        text = item["text"].strip()
        source = "ocr"
        if row == 0:
            text, source = HEADERS[col], "header_geometry"
        elif col == 0:
            text, source = str(row), "sequence_check"
        elif col in (5, 7):
            text = connectors[digest(image_cell(gray, row, col))]
            source = "pixel_identical_connector"
        elif col == 9:
            text = twists[digest(image_cell(gray, row, col))]
            source = "pixel_identical_twist"
        elif col == 3 and row in COLOR_CORRECTIONS:
            text, source = COLOR_CORRECTIONS[row], "image_read_correction"

        if not text:
            raise AssertionError(f"Empty cell: row {row}, column {col}")
        if row:
            if col == 1 and not re.fullmatch(r"\d{2,5}", text):
                raise AssertionError((row, col, text))
            if col == 2 and text not in {"-", "0.35", "0.5", "0.75", "1", "1.5", "2.5"}:
                raise AssertionError((row, col, text))
            if col == 3 and not re.fullmatch(r"-|[A-Z]{2}(?:/[A-Z]{2})?", text):
                raise AssertionError((row, col, text))
            if col == 4 and text not in {"CU-R2P1.4", "CU-R2P1.6", "CU-R2P1.9",
                                         "CU-R2P2.1", "CU-R2P2.4", "CU-R2P3", "COAX 2.8"}:
                raise AssertionError((row, col, text))
            if col == 6 and not re.fullmatch(r"\d{1,2}", text):
                raise AssertionError((row, col, text))
            if col == 8 and not re.fullmatch(r"\d{1,2}|S/J-[12]", text):
                raise AssertionError((row, col, text))
        result.append({**item, "text": text, "decision_source": source})
    if len(result) != 720:
        raise AssertionError("Incomplete table")
    return result


def script_for(cells: list[dict]) -> str:
    commands = [
        '(entmake (list (cons 0 "LAYER") (cons 2 "CIRCUIT_TABLE_GRID") '
        '(cons 70 0) (cons 62 8) (cons 6 "Continuous")))',
        '(entmake (list (cons 0 "LAYER") (cons 2 "CIRCUIT_TABLE_TEXT") '
        '(cons 70 0) (cons 62 7) (cons 6 "Continuous")))',
        '(setq circuit_lines 0 circuit_text 0)',
    ]
    segments = []
    for x in XS:
        segments.append((pixel_to_dwg(x, 0), pixel_to_dwg(x, 1584)))
    for y in range(0, 1585, 22):
        segments.append((pixel_to_dwg(0, y), pixel_to_dwg(1204, y)))
    assert len(segments) == 84
    for (x0, y0), (x1, y1) in segments:
        commands.append(
            f'(if (entmakex (list (cons 0 "LINE") (cons 100 "AcDbEntity") '
            f'(cons 8 "CIRCUIT_TABLE_GRID") (cons 100 "AcDbLine") '
            f'(cons 10 (list {x0:.8f} {y0:.8f} 0.0)) '
            f'(cons 11 (list {x1:.8f} {y1:.8f} 0.0)))) '
            f'(setq circuit_lines (1+ circuit_lines)))'
        )
    for cell in cells:
        row, col = cell["row"], cell["column"]
        x0, y_top = pixel_to_dwg(XS[col], row * 22)
        x1, _ = pixel_to_dwg(XS[col + 1], row * 22)
        x, y = x0 + 0.045, y_top - 0.043
        width = x1 - x0 - 0.08
        commands.append(
            f'(if (entmakex (list (cons 0 "MTEXT") (cons 100 "AcDbEntity") '
            f'(cons 8 "CIRCUIT_TABLE_TEXT") (cons 100 "AcDbMText") '
            f'(cons 10 (list {x:.8f} {y:.8f} 0.0)) (cons 40 0.205) '
            f'(cons 41 {width:.8f}) (cons 71 1) (cons 72 1) '
            f'(cons 1 "{literal(cell["text"])}"))) '
            f'(setq circuit_text (1+ circuit_text)))'
        )
    commands += [
        '(princ (strcat "\\nCIRCUIT_LINES=" (itoa circuit_lines) '
        '" CIRCUIT_TEXT=" (itoa circuit_text)))',
        '_.QSAVE', '_.QUIT', '_N',
    ]
    return "\n".join(commands) + "\n"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not SOURCE.is_file() or not CORE_CONSOLE.is_file():
        raise FileNotFoundError("Corrected DWG or AutoCAD Core Console missing")
    cells = resolved_cells()
    with CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        for row in range(1, 72):
            writer.writerow([cells[row * 10 + col]["text"] for col in range(10)])
    REPORT.write_text(json.dumps({"source_image_bbox_pdf": IMAGE_BBOX,
                                  "rows": 71, "columns": HEADERS, "cells": cells},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(SOURCE, OUTPUT)
    script = OUTPUT.with_suffix(".scr")
    script.write_text(script_for(cells), encoding="ascii")
    log = OUTPUT.with_suffix(".log")
    run_console(OUTPUT, script, log)
    report = log.read_bytes().decode("utf-16-le", errors="replace")
    if "CIRCUIT_LINES=84 CIRCUIT_TEXT=720" not in report:
        raise RuntimeError(f"AutoCAD did not save all circuit table entities: {log}")
    print("DWG", OUTPUT)
    print("CSV", CSV)
    print("TABLE_ROWS=71 CELLS=720 GRID_LINES=84")


if __name__ == "__main__":
    main()
