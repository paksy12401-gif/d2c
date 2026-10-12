"""기존 결합본에 추가 원본 한 장 및 증강/박스 16장을 반영합니다."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from augment_dataset import augment, rgb_image
from add_yolo_boxes import find_box, transform_box
from prepare_cad_captures import write_sample, verify


def run(source, root, class_name):
    raw = source.read_bytes()
    image = rgb_image(raw)
    box = find_box(image)
    classes = json.loads((root / "classes.json").read_text(encoding="utf-8"))
    ids = {name: int(k) for k, name in classes.items()}
    if class_name not in ids:
        raise ValueError("추가 클래스 매핑을 먼저 확정해야 합니다.")
    records = json.loads((root / "index.json").read_text(encoding="utf-8"))
    boxes = json.loads((root / "source_boxes.json").read_text(encoding="utf-8"))
    sid = f"pdf_extra_{class_name}_001"
    if sid in boxes:
        raise ValueError("이 추가 원본은 이미 처리했습니다.")
    backup = root / "source_captures_extra" / class_name / source.name
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(raw)
    base, placed = augment(image, 0, 0.95, False, random.Random(42))
    original_relative = Path("originals") / class_name / source.name
    # 사용자 입력 원본은 유지하고 결합본 내부 사본만 640으로 맞춥니다.
    existing_image = root / "images" / original_relative
    if existing_image.exists() and existing_image.read_bytes() != raw:
        raise ValueError("결합본에 같은 이름의 다른 이미지가 있습니다.")
    sample = write_sample(root, base, original_relative,
                          transform_box(box, image.size, 0, placed), ids[class_name], sid, "original")
    added = [sample]
    boxes[sid] = {**sample, "class_name": class_name, "class_id": ids[class_name],
                  "input_size": list(image.size), "input_bbox_xyxy_px": box,
                  "source_capture": backup.relative_to(root).as_posix(),
                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "method": "nonwhite_foreground_extent", "reviewed": False,
                  "split_group_hint": f"same_symbol_instance:{class_name}"}
    rng = random.Random(42)
    for angle in (0, 90, 180, 270):
        for profile, fraction in enumerate((0.85, 0.95)):
            for gray in (False, True):
                canvas, placed = augment(image, angle, fraction, gray, rng)
                kind = "gray" if gray else "color"
                relative = Path("augmented") / class_name / f"{sid}_r{angle:03d}_s{profile}_{kind}.png"
                added.append(write_sample(root, canvas, relative,
                                          transform_box(box, image.size, angle, placed),
                                          ids[class_name], sid, "augmented", angle, gray))
    verify(root, added, classes)
    records += added
    (root / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "source_boxes.json").write_text(json.dumps(boxes, ensure_ascii=False, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (4 * 260, 2 * 285), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    examples = [r for r in added if r["kind"] == "augmented" and "_s0_" in r["image"]]
    for k, sample in enumerate(examples):
        image = Image.open(root / sample["image"]).convert("RGB")
        ImageDraw.Draw(image).rectangle(sample["bbox_xyxy_px"], outline="red", width=3)
        image.thumbnail((250, 250))
        x, y = k % 4 * 260, k // 4 * 285
        sheet.paste(image, (x + 5, y + 28))
        draw.text((x + 5, y + 5), Path(sample["image"]).stem.replace(sid + "_", ""), fill="black")
    sheet.save(root / "box_previews" / f"pdf_extra_augmented_{class_name}.png")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary.update({"added_pdf_originals": 1, "added_pdf_augmented": 16,
                    "combined_total_labels": len(records), "combined_originals": len(boxes),
                    "combined_augmented": sum(r["kind"] == "augmented" for r in records)})
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "README.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"추가 PDF 캡처 {class_name}: 원본 1장+증강 16장, 박스 포함, 모두 640x640.\n")
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    listed_images = {r["image"] for r in records}
    actual_images = {p.relative_to(root).as_posix() for p in (root / "images").rglob("*.png")}
    assert actual_images == listed_images, (actual_images - listed_images, listed_images - actual_images)
    actual_labels = {p.relative_to(root).as_posix() for p in (root / "labels").rglob("*.txt")}
    assert actual_labels == {r["label"] for r in records}
    zip_path = shutil.make_archive(str(root), "zip", root_dir=root.parent, base_dir=root.name)
    print(json.dumps({"summary": summary, "zip": zip_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--class-name", required=True)
    a = p.parse_args()
    run(a.source.resolve(), a.dataset.resolve(), a.class_name)
