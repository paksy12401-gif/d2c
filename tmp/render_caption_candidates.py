import json
from pathlib import Path

import pymupdf

root = Path(__file__).resolve().parents[1]
pdf = Path(r"C:\Users\Win11Pro\Desktop\1. DWG 객체 변환 도면 생성 샘플\DWG 객체 변환 샘플.pdf")
out = root / "blockified_dwg" / "caption_crops"
out.mkdir(exist_ok=True)
regions = json.loads((root / "blockified_dwg" / "caption_regions_1x.json").read_text(encoding="utf-8"))
regions = [r for r in regions if r[2] >= 150]
with pymupdf.open(pdf) as document:
    page = document[0]
    for index, (x, y, w, h) in enumerate(regions, 1):
        box = pymupdf.Rect(max(0, x - 18), max(0, y - 18), min(5000, x + w + 18), min(2200, y + h + 18))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=box, alpha=False)
        pix.save(out / f"caption_{index:02d}.png")
        regions[index - 1] = {"id": index, "bbox_pdf": [box.x0, box.y0, box.x1, box.y1],
                              "region_pdf": [x, y, x + w, y + h]}
(out / "regions.json").write_text(json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8")
print("crops", len(regions), out)
