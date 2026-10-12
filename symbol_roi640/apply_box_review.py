"""108개 원본의 시각 검수 결과를 적용하고 증강 라벨을 갱신합니다.

이미지는 바꾸지 않습니다. 종류 판별/분할/학습은 이 검수의 범위가 아닙니다.
"""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from add_yolo_boxes import label_line, transform_box

PROJECT = Path(__file__).parent
ROOT = PROJECT / "dataset_yolo26_combined_20261008"
OUT = PROJECT / "dataset_yolo26_boxreviewed_20261008"

# 검사 페이지 전체를 확인한 뒤, 불필요한 외부 표시가 포함된 사례만 수정.
CORRECTIONS = {
    "src_001": "좌측 본체 밖 표시 때문에 넓어진 박스를 본체로 조정",
    "src_009": "상단 이웃 도형의 검은 원호 제외",
    "src_010": "우측 본체 밖 검은 화살표 끝 제외",
    "src_035": "좌측 본체 밖 고립된 작은 표시 제외",
    "src_036": "상단 이웃 도형의 원호 제외",
    "src_037": "상단 이웃 도형의 원호 제외",
    "src_043": "우측 이웃 도형의 검은 원호 제외",
    "src_044": "우측 본체 밖 표시로 생긴 여백 축소",
    "src_045": "상단 본체 밖 검은 표시 제외",
    "src_048": "본체 아래로 나온 축선 끝 제외",
    "src_049": "본체 아래로 나온 축선 끝 제외",
    "src_053": "상단 이웃 도형의 원호 제외",
    "src_054": "상단 원호와 본체 아래 치수 표시로 생긴 여백 제외",
    "src_055": "우측 본체 밖 검은 화살표 끝 제외",
    "src_056": "우측 본체 밖 검은 화살표 끝 제외",
    "src_059": "본체 아래로 길게 나온 축선 제외",
    "src_060": "본체 아래로 길게 나온 축선 제외",
    "src_062": "상단 이웃 도형의 원호 제외",
    "src_063": "우측 본체 밖 검은 축선 끝 제외",
    "pdf_extra_17_001": "우측 본체 밖 화살표 끝 제외; 상단 잘림 가능성은 별도 표시",
}


def source_path(sid, source, root):
    if sid.startswith("cad_"):
        return root / source["source_capture"].replace("source_captures/", "source_captures_g/", 1)
    if sid.startswith("pdf_extra_"):
        return root / source["source_capture"]
    return root / source["image"]


def blue_body_box(image):
    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    # 이번 캡처의 파란 본체만 선택. 검은/녹색/빨간 설명선은 제외.
    mask = ((pixels[:, :, 2] - pixels[:, :, 0] >= 12)
            & (pixels[:, :, 1] - pixels[:, :, 0] >= 5)
            & (pixels[:, :, 0] < 225))
    ys, xs = np.where(mask)
    if len(xs) < 100:
        raise ValueError("이 캡처에는 파란 본체 추출을 적용할 수 없습니다.")
    return [max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2),
            min(image.width, int(xs.max()) + 3), min(image.height, int(ys.max()) + 3)]


def old_input_box(sid, source):
    return source["input_bbox_xyxy_px"] if sid.startswith(("cad_", "pdf_extra_")) else source["bbox_xyxy_px"]


def affine_from_existing(old_box, new_box, raw_size, angle, destination_box):
    width, height = raw_size
    rw, rh = (height, width) if angle in (90, 270) else (width, height)
    old_rotated = transform_box(old_box, raw_size, angle, [0, 0, rw, rh])
    new_rotated = transform_box(new_box, raw_size, angle, [0, 0, rw, rh])
    sx = (destination_box[2] - destination_box[0]) / (old_rotated[2] - old_rotated[0])
    sy = (destination_box[3] - destination_box[1]) / (old_rotated[3] - old_rotated[1])
    return [destination_box[0] + (new_rotated[0] - old_rotated[0]) * sx,
            destination_box[1] + (new_rotated[1] - old_rotated[1]) * sy,
            destination_box[0] + (new_rotated[2] - old_rotated[0]) * sx,
            destination_box[1] + (new_rotated[3] - old_rotated[1]) * sy]


def validate(root, records, sources, classes):
    changed = 0
    for record in records:
        image_path, label_path = root / record["image"], root / record["label"]
        assert image_path.read_bytes() == (ROOT / record["image"]).read_bytes()
        with Image.open(image_path) as image:
            size = image.size
        fields = label_path.read_text().split()
        assert len(fields) == 5
        class_id = int(fields[0])
        assert class_id == sources[record["source_id"]]["class_id"]
        assert str(class_id) in classes
        cx, cy, width, height = map(float, fields[1:])
        assert width > 0 and height > 0
        assert cx - width / 2 >= -1e-7 and cx + width / 2 <= 1 + 1e-7
        assert cy - height / 2 >= -1e-7 and cy + height / 2 <= 1 + 1e-7
        if record["kind"] == "augmented":
            assert size == (640, 640)
        changed += label_path.read_bytes() != (ROOT / record["label"]).read_bytes()
    assert len(list((root / "labels").rglob("*.txt"))) == len(records) == 1836
    return changed


def previews(root, old_sources, sources):
    folder = root / "box_review"
    folder.mkdir(exist_ok=True)
    pairs = list(CORRECTIONS)
    for start in range(0, len(pairs), 8):
        chunk = pairs[start:start + 8]
        sheet = Image.new("RGB", (4 * 300, math.ceil(len(chunk) / 2) * 345), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, sid in enumerate(chunk):
            record = sources[sid]
            for side, source in enumerate((old_sources[sid], record)):
                image = Image.open(source_path(sid, source, root)).convert("RGB")
                box = old_input_box(sid, source)
                ImageDraw.Draw(image).rectangle(box, outline="red" if side == 0 else "green", width=max(2, round(max(image.size) / 180)))
                image.thumbnail((288, 305))
                x, y = (index % 2 * 2 + side) * 300, index // 2 * 345
                sheet.paste(image, (x + (300 - image.width) // 2, y + 30))
                draw.text((x + 6, y + 6), f"{record['class_name']} {sid} {'BEFORE' if side == 0 else 'AFTER'}", fill="black")
        sheet.save(folder / f"corrections_{start // 8 + 1:02d}.png")
    # 검수 후 108개 원본을 모두 다시 표시. CAD 대비 조정은 검수 화면만 적용.
    items = list(sources.items())
    for start in range(0, len(items), 12):
        chunk = items[start:start + 12]
        sheet = Image.new("RGB", (4 * 380, math.ceil(len(chunk) / 4) * 440), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for index, (sid, source) in enumerate(chunk):
            image = Image.open(source_path(sid, source, root)).convert("RGB")
            if sid.startswith("cad_"):
                pixels = np.asarray(image).astype(np.int16)
                background = np.array(source["background_rgb"], dtype=np.int16)
                difference = np.max(np.abs(pixels - background), axis=2)
                image = Image.fromarray(255 - np.clip(difference * 4, 0, 255).astype(np.uint8)).convert("RGB")
            box = old_input_box(sid, source)
            ImageDraw.Draw(image).rectangle(box, outline="green", width=max(2, round(max(image.size) / 230)))
            image.thumbnail((368, 400))
            x, y = index % 4 * 380, index // 4 * 440
            sheet.paste(image, (x + (380 - image.width) // 2, y + 30))
            status = "RECAPTURE?" if source.get("capture_review_status") == "recapture_recommended" else ""
            draw.text((x + 8, y + 6), f"{source['class_name']} {sid} {status}", fill="black")
        sheet.save(folder / f"reviewed_sources_{start // 12 + 1:02d}.png")


def run():
    if OUT.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {OUT}")
    old_sources = json.loads((ROOT / "source_boxes.json").read_text(encoding="utf-8"))
    sources = json.loads((ROOT / "source_boxes.json").read_text(encoding="utf-8"))
    classes = json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))
    records = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    with (PROJECT / "dataset_augmented_20261007" / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = {"images/" + row["image"]: row for row in csv.DictReader(stream)}
    new_boxes = {}
    for sid in CORRECTIONS:
        source = sources[sid]
        image = Image.open(source_path(sid, source, ROOT)).convert("RGB")
        new_boxes[sid] = blue_body_box(image)
        assert new_boxes[sid] != old_input_box(sid, old_sources[sid])
    shutil.copytree(ROOT, OUT)
    for record in records:
        sid = record["source_id"]
        if sid not in new_boxes:
            continue
        source = old_sources[sid]
        new_box = new_boxes[sid]
        if sid.startswith("src_"):
            if record["kind"] == "original":
                box, size = new_box, source["size"]
            else:
                row = manifest[record["image"]]
                box = transform_box(new_box, source["size"], int(row["rotation_ccw"]),
                                    json.loads(row["placed_image_rect_NOT_symbol_bbox"]))
                size = (640, 640)
        else:
            size = (640, 640)
            box = affine_from_existing(source["input_bbox_xyxy_px"], new_box, source["input_size"],
                                       int(record["rotation_ccw"]), record["bbox_xyxy_px"])
        (OUT / record["label"]).write_text(label_line(source["class_id"], box, size), encoding="utf-8")
        record["bbox_xyxy_px"] = box
    originals = {r["source_id"]: r for r in records if r["kind"] == "original"}
    warnings = []
    for sid, source in sources.items():
        source["reviewed"] = True
        source["reviewer"] = "assistant_visual_review"
        source["review_scope"] = "bbox alignment; class identity and complete capture not certified"
        source["review_status"] = "bbox_corrected" if sid in CORRECTIONS else "bbox_alignment_pass"
        source["review_note"] = CORRECTIONS.get(sid, "표시된 구성 형상을 박스가 감싸는지 시각 확인")
        if sid in new_boxes:
            if sid.startswith("src_"):
                source["bbox_xyxy_px"] = new_boxes[sid]
            else:
                source["input_bbox_xyxy_px"] = new_boxes[sid]
                source["bbox_xyxy_px"] = originals[sid]["bbox_xyxy_px"]
        if sid.startswith("cad_"):
            source["source_capture"] = source["source_capture"].replace("source_captures/", "source_captures_g/", 1)
        if sid in ("src_009", "pdf_extra_17_001"):
            source["capture_review_status"] = "recapture_recommended"
            source["capture_review_note"] = ("12번 캡처 하단 구성 형상이 경계에서 끊겨 보임" if sid == "src_009"
                                            else "17번 캡처 상단 형상이 이미지 밖으로 이어져 보임")
            warnings.append({"source_id": sid, "class_name": source["class_name"],
                             "reason": source["capture_review_note"], "image": source["image"]})
        else:
            source["capture_review_status"] = "not_certified_by_bbox_review"
        if source["class_name"] == "24":
            source["review_note"] += "; 사용자 지정 화살표 유지"
        if source["class_name"] == "20+6":
            source["review_note"] += "; 결합 형상 하나의 박스 유지"
        image = Image.open(source_path(sid, source, OUT)).convert("RGB")
        box = old_input_box(sid, source)
        assert 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
        source["capture_boundary_contact"] = (box[0] == 0 or box[1] == 0 or box[2] == image.width or box[3] == image.height)
    changed = validate(OUT, records, sources, classes)
    assert changed == len(CORRECTIONS) * 17 == 340
    assert sources["src_047"]["bbox_xyxy_px"] == old_sources["src_047"]["bbox_xyxy_px"]
    report = {"reviewed_source_images": len(sources), "corrected_source_boxes": len(CORRECTIONS),
              "updated_labels": changed, "total_labels": len(records), "classes": len(classes),
              "images_modified": 0, "corrected_sources": list(CORRECTIONS),
              "recapture_recommendations": warnings,
              "boundary_contact_count": sum(s["capture_boundary_contact"] for s in sources.values()),
              "boundary_contact_note": "경계 접촉만으로 잘린 이미지라고 단정하지 않음",
              "split_assigned": False, "training_performed": False,
              "CAD_source_reference_paths_fixed": 38}
    (OUT / "source_boxes.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "bbox_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary.update({"bbox_visual_review_completed": True, "bbox_corrected_sources": len(CORRECTIONS),
                    "bbox_updated_labels": changed, "automatic_bbox_review_pending": False,
                    "recapture_recommended_sources": [w["source_id"] for w in warnings]})
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    previews(OUT, old_sources, sources)
    # 이전 박스 미리보기는 검수 전 자료임을 명시하여 혼동 방지.
    with (OUT / "README.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n정답박스 시각 검수: 원본 108장 확인, 20장 수정, 라벨 340개 갱신.\n"
                     "box_previews는 검수 전 그림입니다. 검수 후 비교는 box_review를 보세요.\n"
                     "12/src_009, 17/pdf_extra_17_001은 잘림 가능성으로 재캡처 권장합니다.\n"
                     "이미지를 삭제하거나 학습에서 제외하지 않았습니다.\n"
                     "reviewed는 박스 외곽 검수를 뜻하며 독립적인 정답 인증을 뜻하지 않습니다.\n"
                     "학습/검증 분리 및 학습은 아직 실행하지 않았습니다.\n")
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    provenance = OUT / "review_provenance"
    provenance.mkdir()
    shutil.copy2(PROJECT / "dataset_augmented_20261007" / "manifest.csv", provenance / "augmentation_manifest_pdf.csv")
    archive = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
    print(json.dumps({"report": report, "zip": archive}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
