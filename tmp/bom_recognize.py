import json
import os
from pathlib import Path

import cv2
import numpy as np

from models_local import _offline

os.environ["CAD_PADDLE_MODEL_ROOT"] = r"C:\Users\Win11Pro\Documents\CAD_OCR_MODELS"
model_dir = Path(os.environ["CAD_PADDLE_MODEL_ROOT"]) / "official_models" / "korean_PP-OCRv5_mobile_rec"
image = cv2.imread(r"blockified_dwg\bom_full_2x.png")
blue = (image[:, :, 0] > image[:, :, 1] * 1.1) & (image[:, :, 0] > image[:, :, 2] * 1.1) & (image[:, :, 0] > 90)


def lines(profile, threshold):
    indexes = np.flatnonzero(profile > threshold)
    groups = np.split(indexes, np.where(np.diff(indexes) > 1)[0] + 1)
    return [int(np.median(g)) for g in groups if len(g)]


xs = lines(blue.sum(axis=0), 1300)
ys = lines(blue.sum(axis=1), 2500)
assert len(xs) == 6 and len(ys) == 33, (xs, ys)
cells = []
crops = []
for row in range(32):
    for col in range(5):
        x0, x1 = xs[col] + 5, xs[col + 1] - 5
        y0, y1 = ys[row] + 5, ys[row + 1] - 5
        cells.append({"row": 31 - row if row < 31 else "header", "column": col,
                      "bbox_px": [x0, y0, x1, y1]})
        crop = image[y0:y1, x0:x1]
        crops.append(crop)

with _offline(paddle_cache=model_dir.parent.parent):
    from paddleocr import TextRecognition
    model = TextRecognition(model_name="korean_PP-OCRv5_mobile_rec",
                            model_dir=str(model_dir), device="cpu", enable_mkldnn=False)
    print("model ready; cells", len(cells), flush=True)
    for start in range(0, len(cells), 16):
        results = model.predict(input=crops[start:start + 16], batch_size=16)
        for cell, result in zip(cells[start:start + 16], results):
            payload = result.json.get("res", result.json)
            cell["text"] = str(payload.get("rec_text", ""))
            cell["confidence"] = float(payload.get("rec_score", 0))
        print("done", min(start + 16, len(cells)), flush=True)

Path(r"blockified_dwg\bom_cell_ocr.json").write_text(
    json.dumps({"xs": xs, "ys": ys, "cells": cells}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
for cell in cells:
    print(cell["row"], cell["column"], round(cell["confidence"], 3), cell["text"])
