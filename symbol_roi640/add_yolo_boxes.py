"""폴더별 단일 심볼 캡처에 자동 박스를 붙이고 증강 좌표에 전달합니다.

흰 여백을 제외한 전경 범위가 자동 박스입니다. 선/문자/화살표가 함께
들어 있으면 그 내용도 포함될 수 있으므로 source_boxes.json에서 검수합니다.
"""
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def find_box(image):
    # 아주 옅은 안티앨리어싱까지 포함하되 3픽셀 미만의 고립 점만 제외합니다.
    mask = (np.min(np.asarray(image.convert("RGB")), axis=2) < 245).astype(np.uint8)
    count, connected, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    ys, xs = np.where(keep[connected])
    if not len(xs):
        raise ValueError("전경이 없는 이미지입니다.")
    width, height = image.size
    return [max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2),
            min(width, int(xs.max()) + 3), min(height, int(ys.max()) + 3)]


def transform_box(box, size, angle, placed):
    x1, y1, x2, y2 = box
    width, height = size
    if angle == 90:
        box, width, height = [y1, width - x2, y2, width - x1], height, width
    elif angle == 180:
        box = [width - x2, height - y2, width - x1, height - y1]
    elif angle == 270:
        box, width, height = [height - y2, x1, height - y1, x2], height, width
    elif angle != 0:
        raise ValueError("0/90/180/270도만 지원합니다.")
    left, top, right, bottom = placed
    sx, sy = (right - left) / width, (bottom - top) / height
    return [left + box[0] * sx, top + box[1] * sy,
            left + box[2] * sx, top + box[3] * sy]


def label_line(class_id, box, size):
    x1, y1, x2, y2 = box
    width, height = size
    assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    values = [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
              (x2 - x1) / width, (y2 - y1) / height]
    return str(class_id) + " " + " ".join(f"{v:.8f}" for v in values) + "\n"


def run(dataset, output):
    if output.exists():
        raise FileExistsError(f"새 결과 경로를 지정하세요: {output}")
    sources = json.loads((dataset / "sources.json").read_text(encoding="utf-8"))
    classes = json.loads((dataset / "classes.json").read_text(encoding="utf-8"))
    class_ids = {name: int(number) for number, name in classes.items()}
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        variants = list(csv.DictReader(stream))
    source_boxes = {}
    for source in sources:
        path = dataset / "cleaned" / source["class_name"] / (source["source_id"] + ".png")
        with Image.open(path) as image:
            image = image.convert("RGB")
            box = find_box(image)
            source_boxes[source["source_id"]] = {
                "class_name": source["class_name"], "class_id": class_ids[source["class_name"]],
                "image": f"images/originals/{source['class_name']}/{path.name}",
                "size": list(image.size), "bbox_xyxy_px": box,
                "method": "nonwhite_foreground_extent", "reviewed": False,
                "touches_image_edge": box[0] == 0 or box[1] == 0 or box[2] == image.width or box[3] == image.height}

    output.mkdir(parents=True)
    records = []
    for source in sources:
        sid, cname = source["source_id"], source["class_name"]
        record = source_boxes[sid]
        path = dataset / "cleaned" / cname / (sid + ".png")
        image_path = output / record["image"]
        label_path = output / "labels" / "originals" / cname / (sid + ".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, image_path)
        label_path.write_text(label_line(record["class_id"], record["bbox_xyxy_px"], record["size"]), encoding="utf-8")
        records.append({"image": record["image"], "label": label_path.relative_to(output).as_posix(),
                        "source_id": sid, "kind": "original", "split": "UNASSIGNED"})
    for variant in variants:
        sid = variant["source_id"]
        source = source_boxes[sid]
        relative = Path(variant["image"])
        image_path = output / "images" / relative
        label_path = (output / "labels" / relative).with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset / relative, image_path)
        box = transform_box(source["bbox_xyxy_px"], source["size"],
                            int(variant["rotation_ccw"]),
                            json.loads(variant["placed_image_rect_NOT_symbol_bbox"]))
        label_path.write_text(label_line(source["class_id"], box, (640, 640)), encoding="utf-8")
        records.append({"image": image_path.relative_to(output).as_posix(),
                        "label": label_path.relative_to(output).as_posix(),
                        "source_id": sid, "kind": "augmented", "split": "UNASSIGNED"})

    (output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "source_boxes.json").write_text(json.dumps(source_boxes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    # 모든 원본의 박스와 대표 증강본을 이미지로 확인할 수 있습니다.
    preview_dir = output / "box_previews"
    preview_dir.mkdir()
    items = list(source_boxes.items())
    for start in range(0, len(items), 24):
        page_items = items[start:start + 24]
        sheet = Image.new("RGB", (6 * 240, math.ceil(len(page_items) / 6) * 260), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, (sid, record) in enumerate(page_items):
            image = Image.open(output / record["image"]).convert("RGB")
            ImageDraw.Draw(image).rectangle(record["bbox_xyxy_px"], outline="red", width=max(2, round(max(image.size) / 180)))
            image.thumbnail((230, 228))
            x, y = index % 6 * 240, index // 6 * 260
            sheet.paste(image, (x + (240 - image.width) // 2, y + 28))
            draw.text((x + 5, y + 5), f"{record['class_name']} / {sid}", fill="black")
        sheet.save(preview_dir / f"originals_{start // 24 + 1:02d}.png")
    for cname in ("20+6", "13590001"):
        sid = next(k for k, v in source_boxes.items() if v["class_name"] == cname)
        examples = [r for r in records if r["source_id"] == sid and r["kind"] == "augmented" and "_s0_" in r["image"]]
        sheet = Image.new("RGB", (4 * 260, 2 * 285), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, example in enumerate(examples):
            image = Image.open(output / example["image"]).convert("RGB")
            values = list(map(float, (output / example["label"]).read_text().split()[1:]))
            cx, cy, bw, bh = values
            box = [(cx - bw / 2) * 640, (cy - bh / 2) * 640,
                   (cx + bw / 2) * 640, (cy + bh / 2) * 640]
            ImageDraw.Draw(image).rectangle(box, outline="red", width=3)
            image.thumbnail((250, 250))
            x, y = index % 4 * 260, index // 4 * 285
            sheet.paste(image, (x + 5, y + 30))
            draw.text((x + 5, y + 5), Path(example["image"]).stem.replace(sid + "_", ""), fill="black")
        sheet.save(preview_dir / f"augmented_{cname}.png")

    for record in records:
        fields = (output / record["label"]).read_text().split()
        assert len(fields) == 5 and 0 <= int(fields[0]) < len(classes)
        cx, cy, width, height = map(float, fields[1:])
        assert width > 0 and height > 0
        assert cx - width / 2 >= -1e-7 and cx + width / 2 <= 1 + 1e-7
        assert cy - height / 2 >= -1e-7 and cy + height / 2 <= 1 + 1e-7
    summary = {"classes": len(classes), "original_labels": len(sources),
               "augmented_labels": len(variants), "total_labels": len(records),
               "automatic_boxes": True, "manual_review_pending": True,
               "train_val_test_split_assigned": False,
               "source_boxes_touching_capture_edge": [k for k, v in source_boxes.items() if v["touches_image_edge"]]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "YOLO 형식 자동 박스 라벨 데이터\n"
        "images/와 labels/의 하위 경로와 파일명이 서로 대응합니다.\n"
        "라벨: class_id x_center y_center width height (0~1 정규화)\n"
        "20+6은 하나의 클래스로, 이미지마다 박스 1개입니다.\n"
        "박스는 흰 여백을 제외한 전경 범위로 자동 계산했습니다.\n"
        "함께 찍힌 문자/선/화살표도 포함될 수 있어 원본 69장의\n"
        "box_previews와 source_boxes.json을 검수해야 합니다.\n"
        "증강 좌표는 원본 박스의 90도 회전, 크기 및 이동을 반영합니다.\n"
        "split은 미지정입니다. index.json의 source_id와 실제 중복/겹침을\n"
        "기준으로 나누고, train에만 해당 원본의 증강본을 사용하세요.\n"
        "원본과 증강본을 임의로 섞어 val/test에 넣으면 평가가 부풀려집니다.\n"
        "학습은 실행하지 않았습니다.\n", encoding="utf-8")
    shutil.copy2(Path(__file__), output / "add_yolo_boxes.py")
    archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(json.dumps({"summary": summary, "output": str(output), "zip": archive}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset.resolve(), args.output.resolve())
