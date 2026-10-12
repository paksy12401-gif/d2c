"""g 폴더 CAD 캡처를 640x640으로 증강/라벨링하고 기존 데이터와 결합."""
import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from augment_dataset import augment, rgb_image
from add_yolo_boxes import label_line, transform_box


def cad_box(image):
    pixels = np.asarray(image)
    # 화면의 대부분을 차지하는 CAD 배경. 흰 여백 검사 대신 색 차이를 사용.
    background = np.median(pixels.reshape(-1, 3), axis=0).astype(np.int16)
    difference = np.max(np.abs(pixels.astype(np.int16) - background), axis=2)
    mask = (difference > 5).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    ys, xs = np.where(keep[components])
    if not len(xs):
        raise ValueError("배경 외 심볼을 찾지 못했습니다.")
    box = [max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2),
           min(image.width, int(xs.max()) + 3), min(image.height, int(ys.max()) + 3)]
    return box, tuple(int(v) for v in background)


def write_sample(root, canvas, relative, box, class_id, source_id, kind, angle=0, gray=False):
    path = root / "images" / relative
    label = (root / "labels" / relative).with_suffix(".txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    label.write_text(label_line(class_id, box, canvas.size), encoding="utf-8")
    return {"image": path.relative_to(root).as_posix(), "label": label.relative_to(root).as_posix(),
            "source_id": source_id, "kind": kind, "split": "UNASSIGNED",
            "rotation_ccw": angle, "grayscale": gray, "bbox_xyxy_px": box}


def preview(root, sources, records):
    folder = root / "box_previews"
    folder.mkdir(exist_ok=True)
    for start in range(0, len(sources), 24):
        chunk = sources[start:start + 24]
        sheet = Image.new("RGB", (6 * 240, math.ceil(len(chunk) / 6) * 260), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for j, source in enumerate(chunk):
            image = Image.open(root / source["image"]).convert("RGB")
            ImageDraw.Draw(image).rectangle(source["bbox_xyxy_px"], outline="red", width=3)
            image.thumbnail((230, 225))
            x, y = j % 6 * 240, j // 6 * 260
            sheet.paste(image, (x + 5, y + 28))
            draw.text((x + 5, y + 5), f"{source['class_name']} / {source['source_id']}", fill="black")
        sheet.save(folder / f"cad_originals_{start // 24 + 1:02d}.png")
    for cname in ("20+6", "17", "13590001"):
        sid = next(s["source_id"] for s in sources if s["class_name"] == cname)
        examples = [r for r in records if r["source_id"] == sid and r["kind"] == "augmented" and "_s0_" in r["image"]]
        sheet = Image.new("RGB", (4 * 260, 2 * 285), "#dddddd")
        draw = ImageDraw.Draw(sheet)
        for j, sample in enumerate(examples):
            image = Image.open(root / sample["image"]).convert("RGB")
            ImageDraw.Draw(image).rectangle(sample["bbox_xyxy_px"], outline="red", width=3)
            image.thumbnail((250, 250))
            x, y = j % 4 * 260, j // 4 * 285
            sheet.paste(image, (x + 5, y + 28))
            draw.text((x + 5, y + 5), Path(sample["image"]).stem.replace(sid + "_", ""), fill="black")
        sheet.save(folder / f"cad_augmented_{cname}.png")


def verify(root, records, classes):
    gray_count = 0
    for record in records:
        with Image.open(root / record["image"]) as image:
            assert image.size == (640, 640) and image.mode == "RGB"
            if record.get("grayscale"):
                a = np.asarray(image)
                assert np.array_equal(a[:, :, 0], a[:, :, 1]) and np.array_equal(a[:, :, 1], a[:, :, 2])
                gray_count += 1
        fields = (root / record["label"]).read_text().split()
        assert len(fields) == 5 and 0 <= int(fields[0]) < len(classes)
        x, y, w, h = map(float, fields[1:])
        assert w > 0 and h > 0
        assert x - w / 2 >= -1e-7 and x + w / 2 <= 1 + 1e-7
        assert y - h / 2 >= -1e-7 and y + h / 2 <= 1 + 1e-7
    return gray_count


def run(source, existing, output, combined):
    if output.exists() or combined.exists():
        raise FileExistsError("결과 폴더가 이미 있습니다. 새 경로를 지정하세요.")
    files = sorted(source.glob("*/*.png"))
    if not files:
        raise ValueError("CAD 캡처를 찾지 못했습니다.")
    classes = json.loads((existing / "classes.json").read_text(encoding="utf-8"))
    old_classes = classes.copy()
    for name in sorted({f.parent.name for f in files} - set(classes.values())):
        classes[str(len(classes))] = name
    ids = {name: int(number) for number, name in classes.items()}
    # 입력 전부 검사 후 새 파일 작성 시작.
    inputs = [(f, f.read_bytes()) for f in files]
    for f, raw in inputs:
        rgb_image(raw).load()
    output.mkdir(parents=True)
    rng = random.Random(42)
    records, sources = [], []
    for number, (path, raw) in enumerate(inputs, 1):
        cname, sid = path.parent.name, f"cad_{number:03d}"
        image = rgb_image(raw)
        box, background = cad_box(image)
        backup = output / "source_captures" / cname / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(raw)
        # 학습용 원본도 640x640. 입력 원본 자체는 source_captures에 보존.
        base, placed = augment(image, 0, 0.95, False, random.Random(42), background)
        base_box = transform_box(box, image.size, 0, placed)
        sample = write_sample(output, base, Path("originals") / cname / f"{sid}.png",
                              base_box, ids[cname], sid, "original")
        records.append(sample)
        sources.append({**sample, "class_name": cname, "class_id": ids[cname],
                        "source_capture": backup.relative_to(output).as_posix(),
                        "input_size": list(image.size), "input_bbox_xyxy_px": box,
                        "background_rgb": list(background),
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "method": "CAD_background_color_difference_gt5", "reviewed": False,
                        "split_group_hint": f"same_symbol_instance:{cname}"})
        for angle in (0, 90, 180, 270):
            for profile, fraction in enumerate((0.85, 0.95)):
                for gray in (False, True):
                    canvas, placed = augment(image, angle, fraction, gray, rng, background)
                    variant_box = transform_box(box, image.size, angle, placed)
                    kind = "gray" if gray else "color"
                    relative = Path("augmented") / cname / f"{sid}_r{angle:03d}_s{profile}_{kind}.png"
                    records.append(write_sample(output, canvas, relative, variant_box,
                                                ids[cname], sid, "augmented", angle, gray))
    assert len(records) == len(inputs) * 17
    gray_count = verify(output, records, classes)
    preview(output, sources, records)
    (output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "source_boxes.json").write_text(json.dumps({s["source_id"]: s for s in sources}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"classes": len(classes), "cad_source_images": len(inputs),
               "cad_originals_640": len(inputs), "cad_augmented_640": len(inputs) * 16,
               "cad_grayscale_images": gray_count, "cad_total_labels": len(records),
               "original_class_ids_preserved": old_classes == {k: classes[k] for k in old_classes},
               "new_classes": {k: v for k, v in classes.items() if k not in old_classes},
               "automatic_bbox_review_pending": True, "split_assigned": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = (
        "g 폴더 CAD 캡처의 로컬 증강 및 YOLO 라벨\n"
        "학습용 원본/증강본 모두 640x640, 비율 유지, CAD 배경색 패딩\n"
        "source_captures는 입력 크기의 원본 보관이며 학습 입력이 아닙니다.\n"
        "0/90/180/270도 x 크기 2개 x 컬러/그레이스케일 = 원본당 16장\n"
        "박스는 배경색과 다른 전경 외곽의 자동 후보입니다. 검수가 필요합니다.\n"
        "20+6은 하나의 클래스/박스로 유지합니다. 기존 클래스 ID를 유지하고 17을 끝에 추가했습니다.\n"
        "index.json의 split은 미지정입니다. 같은 심볼의 CAD/PDF 캡처와\n"
        "그 증강본은 같은 분할 묶음으로 취급하세요. 증강본은 train에만 사용하세요.\n"
        "학습은 실행하지 않았습니다.\n")
    (output / "README.txt").write_text(readme, encoding="utf-8")
    for name in ("prepare_cad_captures.py", "augment_dataset.py", "add_yolo_boxes.py"):
        shutil.copy2(Path(__file__).parent / name, output / name)

    # 기존 데이터와 한 프로젝트에서 사용할 수 있는 결합본. 사용자 g 원본은 그대로 둠.
    combined.mkdir()
    for name in ("images", "labels"):
        shutil.copytree(existing / name, combined / name, ignore=shutil.ignore_patterns("g"))
        shutil.copytree(output / name, combined / name, dirs_exist_ok=True)
    shutil.copytree(output / "source_captures", combined / "source_captures_g")
    shutil.copytree(existing / "box_previews", combined / "box_previews")
    shutil.copytree(output / "box_previews", combined / "box_previews", dirs_exist_ok=True)
    old_records = json.loads((existing / "index.json").read_text(encoding="utf-8"))
    old_boxes = json.loads((existing / "source_boxes.json").read_text(encoding="utf-8"))
    assert not set(old_boxes).intersection(s["source_id"] for s in sources)
    all_records = old_records + records
    all_boxes = {**old_boxes, **{s["source_id"]: s for s in sources}}
    (combined / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (combined / "index.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (combined / "source_boxes.json").write_text(json.dumps(all_boxes, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in old_records:
        assert (existing / record["label"]).read_bytes() == (combined / record["label"]).read_bytes()
    for record in all_records:
        assert (combined / record["image"]).is_file() and (combined / record["label"]).is_file()
    summary["combined_total_labels"] = len(all_records)
    summary["combined_originals"] = len(old_boxes) + len(sources)
    summary["combined_augmented"] = sum(r["kind"] == "augmented" for r in all_records)
    (combined / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (combined / "README.txt").write_text(readme + "이 결합본은 기존 PDF 캡처 69장/증강 1104장도 포함합니다.\n기존 원본 69장은 입력 해상도를 유지하고, 모든 증강본과 g 학습용 이미지는 640x640입니다.\n", encoding="utf-8")
    cad_zip = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    combined_zip = shutil.make_archive(str(combined), "zip", root_dir=combined.parent, base_dir=combined.name)
    print(json.dumps({"summary": summary, "cad_zip": cad_zip, "combined_zip": combined_zip}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    args = parser.parse_args()
    run(*(getattr(args, name).resolve() for name in ("source", "existing", "output", "combined")))
