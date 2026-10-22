"""Fine-tune the existing drawing model for symbol 1 only.

The source contains one original class-1 crop plus derived views and tiles from
the same drawing. Validation here is a regression check, not independent accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "symbol_roi640" / "dataset_yolo26_all38_eval_20261008"
BASE_WEIGHTS = (ROOT / "symbol_roi640" / "training_yolo26_noaug_20261008"
                / "known_drawing_fit20_durable" / "weights" / "best.pt")
TARGET = ROOT / "symbol_1_finetune"
CLASSES = json.loads((SOURCE / "index.json").read_text(encoding="utf-8"))


def prepare() -> Path:
    random.seed(41)
    positives = []
    negative_by_class: dict[str, list[dict]] = {}
    for item in CLASSES:
        if item.get("split") != "train":
            continue
        if not (SOURCE / item["image"]).is_file():
            continue
        lines = (SOURCE / item["label"]).read_text(encoding="utf-8").splitlines()
        if any(line.startswith("0 ") for line in lines):
            positives.append(item)
        elif item.get("kind") in {"original", "augmented"}:
            negative_by_class.setdefault(str(item.get("source_id", "")), []).append(item)

    if len(positives) < 10:
        raise ValueError(f"Expected class-1 examples; found {len(positives)}")
    # This folder currently retains the isolated examples, but some older scene
    # paths in index.json are missing on disk. Hold out two derived views.
    positives.sort(key=lambda item: item["image"])
    validation_keys = {item["image"] for item in positives[-2:]}
    negatives = []
    for group in negative_by_class.values():
        random.shuffle(group)
        negatives.extend(group[:2])
    random.shuffle(negatives)
    negatives = negatives[:74]
    validation_negatives = {item["image"] for item in negatives[:8]}

    dataset = TARGET / "dataset"
    manifest = []
    for item in positives + negatives:
        positive = item in positives
        split = "val" if item["image"] in validation_keys | validation_negatives else "train"
        stem = Path(item["image"]).stem
        name = f"{split}_{stem}"
        image_out = dataset / "images" / split / f"{name}.png"
        label_out = dataset / "labels" / split / f"{name}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / item["image"], image_out)
        labels = (SOURCE / item["label"]).read_text(encoding="utf-8").splitlines()
        class_one = [line for line in labels if line.startswith("0 ")] if positive else []
        label_out.write_text("\n".join(class_one) + ("\n" if class_one else ""), encoding="utf-8")
        manifest.append({"image": str(image_out.relative_to(TARGET)), "source": item["image"],
                         "split": split, "class_1_boxes": len(class_one)})

    # Retain the 38-class detection head; only class 0 receives positive labels.
    names = YOLO(str(BASE_WEIGHTS)).names
    yaml_lines = [f'path: "{dataset.as_posix()}"', "train: images/train", "val: images/val",
                  "names:"] + [f'  {index}: "{name}"' for index, name in names.items()]
    config = TARGET / "symbol_1.yaml"
    config.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (TARGET / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"class 1: {len(positives)} positive images, {len(negatives)} negatives", flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if not BASE_WEIGHTS.is_file():
        raise FileNotFoundError(BASE_WEIGHTS)
    config = prepare()
    model = YOLO(str(BASE_WEIGHTS))
    model.train(data=str(config), epochs=args.epochs, imgsz=640, batch=8, device=0,
                workers=0, project=str(TARGET), name="class_1", exist_ok=True,
                optimizer="AdamW", lr0=0.0001, lrf=0.05, seed=41,
                mosaic=0, mixup=0, cutmix=0, copy_paste=0,
                degrees=0, translate=0, scale=0, shear=0, perspective=0,
                fliplr=0, flipud=0, hsv_h=0, hsv_s=0, hsv_v=0,
                erasing=0, plots=True)


if __name__ == "__main__":
    main()
