"""Offline OCR of the 71-row raster circuit table embedded in the source PDF."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocr_audit import _offline, load_recognizer  # noqa: E402

IMAGE = ROOT / "blockified_dwg/right_circuit_source.png"
OUTPUT = ROOT / "blockified_dwg/right_circuit_ocr_raw.json"
XS = [0, 72, 140, 186, 251, 346, 628, 692, 974, 1070, 1204]
HEADERS = ["NO", "WIRE", "SQUA", "COL", "MAT", "FROM_CONNECTOR",
           "PIN ID1", "TO_CONNECTOR", "PIN ID2", "TWIST & SHIELD"]


def main() -> None:
    gray = cv2.imdecode(np.fromfile(IMAGE, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert gray.shape == (1584, 1204)
    assert np.all(gray[np.arange(0, 1584, 22), :].max(axis=1) == 0)
    crops = []
    addresses = []
    for row in range(72):
        for col in range(10):
            x0, x1 = XS[col], XS[col + 1]
            y0, y1 = row * 22, (row + 1) * 22
            cell = gray[y0 + 2:y1 - 1, x0 + 2:x1 - 1]
            if col in (1, 2, 3, 4, 5, 7, 9):
                # Keep table text left-to-right; a small white border improves
                # recognition of the first/last glyph against grid lines.
                cell = cv2.copyMakeBorder(cell, 5, 5, 6, 6,
                                          cv2.BORDER_CONSTANT, value=255)
            scale = 48 / cell.shape[0]
            cell = cv2.resize(cell, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
            crops.append(cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR))
            addresses.append((row, col))
    reader = load_recognizer()
    with _offline(paddle_cache=getattr(reader, "_cad_model_root", None)):
        predictions = reader.predict(input=crops, batch_size=32)
    result = []
    for (row, col), prediction in zip(addresses, predictions):
        payload = prediction.json.get("res", prediction.json)
        result.append({"row": row, "column": col, "header": HEADERS[col],
                       "text": str(payload.get("rec_text", "")).strip(),
                       "confidence": float(payload.get("rec_score", 0.0)),
                       "bbox_px": [XS[col], row * 22, XS[col + 1], (row + 1) * 22]})
    assert len(result) == 720
    OUTPUT.write_text(json.dumps({"image": str(IMAGE), "cells": result},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    for row in (0, 1, 2, 3, 19, 67, 71):
        print(row, " | ".join(result[row * 10 + c]["text"] for c in range(10)))
    print("output", OUTPUT)


if __name__ == "__main__":
    main()
