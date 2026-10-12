"""클래스별 이미지 폴더를 유지하며 640×640 PNG로 저장합니다.

실행: python resize_dataset_640.py
설치: python -m pip install Pillow
INPUT_DIR을 실제 dataset 폴더 경로로 수정하세요.
이미지만 처리합니다. 증강, 라벨 생성, 학습은 실행하지 않습니다.
"""
from pathlib import Path
import csv

from PIL import Image, ImageOps


# 기본값: 이 파이썬 파일 옆의 dataset 폴더.
# 다른 위치라면 Path(r"D:\데이터\dataset")처럼 바꾸세요.
INPUT_DIR = Path(__file__).resolve().parent / "dataset"
OUTPUT_DIR = INPUT_DIR.with_name(INPUT_DIR.name + "_640")
SIZE = 640
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def resize_image(path):
    """비율 유지 축소 + 중앙 배치. 작은 원본은 확대하지 않습니다."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGBA")
    original_width, original_height = image.size

    # 투명 배경은 흰색으로 합칩니다. 색상과 명암은 그대로 유지합니다.
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    image = background.convert("RGB")
    image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    width, height = image.size
    left, top = (SIZE - width) // 2, (SIZE - height) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), "white")
    canvas.paste(image, (left, top))

    transform = {
        "original_width": original_width, "original_height": original_height,
        "resized_width": width, "resized_height": height,
        "scale_x": width / original_width, "scale_y": height / original_height,
        "padding_left": left, "padding_top": top,
    }
    return canvas, transform


def main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"INPUT_DIR을 실제 dataset 폴더로 수정하세요: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("결과 폴더는 원본 dataset 폴더 밖에 지정하세요.")
    if output_dir.exists():
        raise FileExistsError(f"덮어쓰지 않습니다. OUTPUT_DIR을 새 폴더로 바꾸세요: {output_dir}")

    files = sorted(p for p in input_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError("이미지를 찾을 수 없습니다.")
    destinations = set()
    for path in files:
        relative = path.relative_to(input_dir)
        if len(relative.parts) < 2:
            raise ValueError(f"이미지를 클래스명 폴더 안에 넣으세요: {relative}")
        # 기존 정답 좌표를 잘못된 위치에 남기는 것을 방지합니다.
        if path.with_suffix(".txt").exists():
            raise ValueError(f"라벨 파일이 있습니다. 이미지와 박스 좌표를 함께 변환해야 합니다: {path}")
        destination = str(relative.with_suffix(".png")).casefold()
        if destination in destinations:
            raise ValueError(f"PNG 변환 후 파일명이 겹칩니다. 원본 파일명을 구분하세요: {relative}")
        destinations.add(destination)

    output_dir.mkdir(parents=True)
    report = output_dir / "resize_manifest.csv"
    with report.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = None
        for number, path in enumerate(files, 1):
            relative = path.relative_to(input_dir)
            try:
                image, transform = resize_image(path)
            except Exception as error:
                raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}") from error
            destination = output_dir / relative.with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, format="PNG")
            row = {"class_name": relative.parts[0], "source": str(relative),
                   "output": str(destination.relative_to(output_dir)), **transform}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            print(f"[{number}/{len(files)}] {relative} → 640×640", flush=True)
    print(f"완료: {len(files)}장\n저장 폴더: {output_dir}\n변환 기록: {report}")


if __name__ == "__main__":
    main()
