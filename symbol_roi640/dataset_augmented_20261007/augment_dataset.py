"""회사 도면을 외부로 보내지 않고 폴더별 심볼 캡처를 증강합니다.

폴더명은 클래스 이름입니다. 생성하는 manifest의 placed_image_rect는
캡처 이미지의 배치 영역이며, 심볼 정답박스가 아닙니다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

SIZE = 640
SEED = 42


def rgb_image(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    return background.convert("RGB")


def augment(image: Image.Image, angle: int, fraction: float, gray: bool, rng):
    """정확한 90도 회전 후 비율 유지 축소/확대 및 흰 배경 내 이동."""
    rotations = {90: Image.Transpose.ROTATE_90,
                 180: Image.Transpose.ROTATE_180,
                 270: Image.Transpose.ROTATE_270}
    if angle:
        image = image.transpose(rotations[angle])
    if gray:
        image = ImageOps.grayscale(image).convert("RGB")
    ratio = SIZE * fraction / max(image.size)
    width, height = [max(1, round(v * ratio)) for v in image.size]
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    # 최대 2% 이동. 모든 캡처 내용이 캔버스 안에 남도록 제한합니다.
    x = max(0, min(SIZE - width, (SIZE - width) // 2 + rng.randint(-12, 12)))
    y = max(0, min(SIZE - height, (SIZE - height) // 2 + rng.randint(-12, 12)))
    canvas = Image.new("RGB", (SIZE, SIZE), "white")
    canvas.paste(image, (x, y))
    return canvas, [x, y, x + width, y + height]


def run(source_zip: Path, replacement: Path, output: Path):
    # 기존 결과를 덮어쓰거나 삭제하지 않습니다.
    if output.exists():
        raise FileExistsError(f"결과 폴더가 이미 있습니다. 새 경로를 지정하세요: {output}")
    replacement_bytes = replacement.read_bytes()
    rgb_image(replacement_bytes)  # 쓰기 전에 교체 이미지 유효성 확인
    originals = []
    removed = []
    replaced = []
    with zipfile.ZipFile(source_zip) as archive:
        entries = sorted((v for v in archive.infolist()
                          if not v.is_dir() and v.filename.lower().endswith(".png")),
                         key=lambda v: v.filename)
        for entry in entries:
            parts = entry.filename.split("/")
            if len(parts) != 3 or parts[0] != "dataset":
                raise ValueError(f"예상하지 못한 폴더 구조: {entry.filename}")
            class_name = parts[1]
            raw = archive.read(entry)
            image = rgb_image(raw)
            if class_name == "11" and image.size == (149, 148):
                removed.append(entry.filename)
                continue
            if class_name == "13590001":
                replaced.append(entry.filename)
                raw = replacement_bytes
                image = rgb_image(raw)
            originals.append((class_name, entry.filename, raw, image))
    if len(removed) != 1 or len(replaced) != 1:
        raise ValueError("11번 작은 이미지 1장과 13590001 이미지 1장을 확인할 수 없습니다.")

    output.mkdir(parents=True)
    classes = sorted({v[0] for v in originals})
    (output / "classes.json").write_text(
        json.dumps(dict(enumerate(classes)), ensure_ascii=False, indent=2), encoding="utf-8")
    rng = random.Random(SEED)
    rows = []
    source_rows = []
    preview_items = []
    pixel_hashes = set()
    per_class = Counter()
    for number, (class_name, original_name, raw, image) in enumerate(originals, 1):
        source_id = f"src_{number:03d}"
        clean_path = output / "cleaned" / class_name / f"{source_id}.png"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_bytes(raw)
        source_rows.append({"source_id": source_id, "class_name": class_name,
                            "original_zip_entry": original_name,
                            "source_sha256": hashlib.sha256(raw).hexdigest(),
                            "size": list(image.size),
                            "replacement": class_name == "13590001"})
        destination = output / "augmented" / class_name
        destination.mkdir(parents=True, exist_ok=True)
        for angle in (0, 90, 180, 270):
            for profile, fraction in enumerate((0.85, 0.95)):
                for gray in (False, True):
                    canvas, rect = augment(image, angle, fraction, gray, rng)
                    kind = "gray" if gray else "color"
                    name = f"{source_id}_r{angle:03d}_s{profile}_{kind}.png"
                    path = destination / name
                    canvas.save(path)
                    pixel_hashes.add(hashlib.sha256(canvas.tobytes()).hexdigest())
                    rows.append({"image": path.relative_to(output).as_posix(),
                                 "source_id": source_id, "class_name": class_name,
                                 "rotation_ccw": angle, "grayscale": gray,
                                 "canvas_fraction": fraction,
                                 "placed_image_rect_NOT_symbol_bbox": json.dumps(rect),
                                 "split": "UNASSIGNED"})
                    per_class[class_name] += 1
                    if class_name in ("20+6", "13590001") and profile == 0:
                        preview_items.append((class_name, angle, kind, canvas.copy()))

    with (output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "sources.json").write_text(
        json.dumps(source_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"source_images": len(originals), "classes": len(classes),
               "augmented_images": len(rows), "grayscale_images": sum(v["grayscale"] for v in rows),
               "unique_augmented_pixel_hashes": len(pixel_hashes),
               "canvas_size": [SIZE, SIZE], "seed": SEED,
               "removed": removed, "replaced": replaced,
               "class_24": "unchanged; arrow retained",
               "combined_class_20+6": "one symbol / one future CAD block",
               "per_class": dict(per_class), "detection_bbox_labels_created": False,
               "split_assigned": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sheet = Image.new("RGB", (8 * 160, 2 * 190), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, (class_name, angle, kind, image) in enumerate(preview_items):
        image.thumbnail((154, 154))
        x, y = index % 8 * 160, index // 8 * 190
        sheet.paste(image, (x + 3, y + 28))
        draw.text((x + 3, y + 5), f"{class_name} {angle} {kind}", fill="black")
    sheet.save(output / "augmentation_preview.png")
    (output / "README.txt").write_text(
        "로컬 심볼 증강 결과\n"
        "cleaned: 11번 149x148 제외, 13590001 교체 후 원본 69장\n"
        "augmented: 각 원본당 16장, 640x640 흰 배경, 비율 유지\n"
        "회전 0/90/180/270도 x 크기 2개 x 컬러/그레이스케일\n"
        "약한 위치 이동 적용. 좌우/상하 반전, 노이즈, 흐림 추가 없음\n"
        "24번 화살표 및 나머지 형상/표시 유지\n"
        "폴더명은 클래스이며 정답박스 좌표는 아직 없습니다.\n"
        "placed_image_rect는 캡처의 배치 범위이며 심볼 박스가 아닙니다.\n"
        "증강본은 원본 source_id로 연결됩니다. 같은 원본과 중복/겹치는\n"
        "캡처는 같은 split에 배정하고, 학습에는 train 원본의 증강본만\n"
        "사용하세요. val/test는 원본으로 평가하세요. split은 미지정입니다.\n"
        "흑백 원본의 grayscale 변환은 새로운 색상 정보를 만들지 않습니다.\n"
        "학습/검출/DWG 생성은 이번 증강 작업에 포함되지 않습니다.\n",
        encoding="utf-8")
    shutil.copy2(Path(__file__), output / "augment_dataset.py")
    # 저장된 이미지와 변경 내용을 다시 확인합니다.
    actual = list((output / "augmented").glob("*/*.png"))
    assert len(actual) == len(originals) * 16 == len(rows)
    for path in actual:
        with Image.open(path) as check:
            assert check.size == (SIZE, SIZE) and check.mode == "RGB"
            check.verify()
    assert per_class["20+6"] == 32
    assert len(list((output / "cleaned" / "11").glob("*.png"))) == 2
    clean_replacement = next((output / "cleaned" / "13590001").glob("*.png"))
    assert clean_replacement.read_bytes() == replacement_bytes
    archive_path = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(json.dumps({"summary": summary, "output": str(output), "zip": archive_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_zip.resolve(), args.replacement.resolve(), args.output.resolve())
