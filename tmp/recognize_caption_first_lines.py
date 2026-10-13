import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_audit import load_recognizer, _offline

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "blockified_dwg/caption_crops"
regions = {r["id"]: r for r in json.loads((BASE / "regions.json").read_text(encoding="utf-8"))}
tops = {1: 282.9, 2: 787.9, 5: 973.9, 6: 1067.5,
        8: 1312.9, 9: 1371.1, 10: 1394.1,
        14: 2126.7, 15: 2127.4, 16: 2127.9}
crops = []
for ident, top in tops.items():
    image = cv2.imdecode(np.fromfile(BASE / f"caption_{ident:02d}.png", dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    offset = regions[ident]["bbox_pdf"][1]
    y0 = max(0, round((top - offset - 2) * 3))
    y1 = min(image.shape[0], round((top - offset + 13) * 3))
    crop = image[y0:y1, :]
    crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    crops.append(crop)
reader = load_recognizer()
with _offline(paddle_cache=getattr(reader, "_cad_model_root", None)):
    results = reader.predict(input=crops, batch_size=10)
out = []
for ident, result in zip(tops, results):
    data = result.json.get("res", result.json)
    out.append({"crop": ident, "text": data.get("rec_text"), "score": data.get("rec_score")})
print(json.dumps(out, ensure_ascii=False, indent=2))
(BASE / "first_line_ocr.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
