"""검수 화면만 렌더링. 학습 이미지/라벨은 변경하지 않습니다."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent / "dataset_yolo26_combined_20261008"
OUT = Path(__file__).parent / "box_review_20261008"


def raw_source(sid, record):
    if sid.startswith("cad_"):
        path = ROOT / record["source_capture"].replace("source_captures/", "source_captures_g/", 1)
        return path, record["input_bbox_xyxy_px"]
    if sid.startswith("pdf_extra_"):
        return ROOT / record["source_capture"], record["input_bbox_xyxy_px"]
    return ROOT / record["image"], record["bbox_xyxy_px"]


def view_image(sid, record):
    path, box = raw_source(sid, record)
    image = Image.open(path).convert("RGB")
    if sid.startswith("cad_"):
        pixels = np.asarray(image).astype(np.int16)
        background = np.array(record["background_rgb"], dtype=np.int16)
        difference = np.max(np.abs(pixels - background), axis=2)
        # 어두운 CAD 선을 검수 화면에서만 밝은 배경 위에 보여 줍니다.
        gray = 255 - np.clip(difference * 4, 0, 255).astype(np.uint8)
        image = Image.fromarray(gray).convert("RGB")
    return image, box


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    records = json.loads((ROOT / "source_boxes.json").read_text(encoding="utf-8"))
    items = list(records.items())
    for start in range(0, len(items), 12):
        chunk = items[start:start + 12]
        sheet = Image.new("RGB", (4 * 380, math.ceil(len(chunk) / 4) * 440), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for j, (sid, record) in enumerate(chunk):
            image, box = view_image(sid, record)
            ImageDraw.Draw(image).rectangle(box, outline="red", width=max(2, round(max(image.size) / 230)))
            image.thumbnail((368, 400))
            x, y = j % 4 * 380, j // 4 * 440
            sheet.paste(image, (x + (380 - image.width) // 2, y + 30))
            draw.text((x + 8, y + 6), f"{record['class_name']} / {sid}", fill="black")
        sheet.save(OUT / f"inspection_{start // 12 + 1:02d}.png")
    print(f"{len(items)} sources, {math.ceil(len(items) / 12)} inspection sheets: {OUT}")
