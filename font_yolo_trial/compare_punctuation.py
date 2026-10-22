"""Compare raw 62/67-class model outputs on fixed user images; no word correction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from evaluate_probes import nms, selfcheck
from prepare import read_image, write_image
from train import overlap, read_labels, read_rows

ROOT = Path(__file__).resolve().parent
CONFIDENCE, IOU = .10, .5


def tiled(model, source, path):
    height, width = source.shape[:2]
    tiles, origins = [], []
    for y in range(0, height, 288):
        for x in range(0, width, 288):
            crop = source[y:y+384, x:x+384]
            tile = np.full((384, 384, 3), 255, dtype=np.uint8)
            tile[:crop.shape[0], :crop.shape[1]] = crop
            tiles.append(tile)
            origins.append((x, y, crop.shape[1], crop.shape[0]))
    predictions = model.predict(tiles, imgsz=384, end2end=False, conf=CONFIDENCE,
                                iou=IOU, agnostic_nms=True, device=0, batch=8, verbose=False)
    assert len(predictions) == len(origins)
    boxes = []
    for prediction, (x, y, w, h) in zip(predictions, origins):
        for box in prediction.boxes.data.cpu().tolist():
            left, top = (48 if x else 0), (48 if y else 0)
            right, bottom = (336 if x+384 < width else w), (336 if y+384 < height else h)
            if not (left <= (box[0]+box[2])/2 < right and top <= (box[1]+box[3])/2 < bottom):
                continue
            box[:4] = [max(x, box[0]+x), max(y, box[1]+y), min(x+w, box[2]+x), min(y+h, box[3]+y)]
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
    merged = np.asarray(nms(boxes, IOU), dtype=np.float32).reshape(-1, 6)
    assert all(0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height for b in merged)
    return Results(orig_img=source, path=str(path), names=model.names, boxes=merged)


def regression(model):
    """Both models see the identical existing synthetic alphanumeric test set."""
    paths = sorted((ROOT / "alphanumeric/images/test").glob("*.png"))
    counts = np.zeros(3, dtype=int)
    predictions = model.predict([str(p) for p in paths], imgsz=384, conf=CONFIDENCE,
                                iou=IOU, end2end=False, device=0, batch=8,
                                stream=True, verbose=False)
    seen = 0
    for i, result in enumerate(predictions):
        seen += 1
        h, w = result.orig_shape
        truth = read_labels(ROOT / "alphanumeric/labels/test" / (paths[i].stem + ".txt"), w, h)
        available = set(range(len(truth)))
        for box in sorted(result.boxes.data.cpu().numpy(), key=lambda b: -b[4]):
            candidate = max(((overlap(box[:4], truth[j][1]), j) for j in available
                             if int(box[5]) == truth[j][0]), default=(0, -1))
            if candidate[0] >= .5:
                available.remove(candidate[1])
                counts[0] += 1
            else:
                counts[1] += 1
        counts[2] += len(available)
    assert seen == len(paths) == 130
    tp, fp, fn = counts.tolist()
    return {"scope": "Same-template synthetic regression; no punctuation ground truth in this older set.",
            "images": seen, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp/max(tp+fp, 1), "recall": tp/max(tp+fn, 1)}


def main():
    selfcheck()
    torch.set_num_threads(4)
    output = ROOT / "punctuation_fixed/user_comparison"
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "punctuation_640": (ROOT / "punctuation_fixed/source_punctuation.png", 640),
        "punctuation_384": (ROOT / "punctuation_fixed/source_punctuation.png", 384),
        "cad_full_640": (ROOT / "alphanumeric/sample_23476557/source.png", 640),
        "cad_tiles_384": (ROOT / "alphanumeric/sample_23476557/source.png", None),
        "compact_384": (ROOT / "alphanumeric/user_comparison/input_original.png", 384),
        "digits_640": (ROOT / "alphanumeric/source_digits.png", 640),
    }
    report = {"scope": "Diagnostic previously seen images. Glyph source strips are training sources; not unseen accuracy.",
              "confidence_threshold": CONFIDENCE, "iou": IOU, "end2end": False,
              "word_correction": False, "models": {}, "inputs": {}, "predictions": [], "synthetic_regression": {}}
    for name, (path, size) in inputs.items():
        report["inputs"][name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "imgsz": size, "tile_settings": {"size": 384, "stride": 288, "agnostic_nms": True} if size is None else None}
    for label, path in {"old62": ROOT / "alphanumeric/training/weights/best.pt",
                        "new67": ROOT / "punctuation_fixed/training/weights/best.pt"}.items():
        model = YOLO(str(path))
        report["models"][label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "classes": model.names}
        for name, (source_path, size) in inputs.items():
            source = read_image(source_path)
            result = tiled(model, source, source_path) if size is None else model.predict(
                source, imgsz=size, rect=True, conf=CONFIDENCE, iou=IOU,
                end2end=False, agnostic_nms=False, device=0, verbose=False)[0]
            annotated = output / f"{label}_{name}.png"
            write_image(annotated, result.plot(line_width=1, font_size=10))
            boxes = result.boxes.cpu().numpy().data.tolist()
            item = {"model": label, "input": name, "raw_lines": read_rows(result),
                    "left_to_right_labels": [result.names[int(b[5])] for b in sorted(boxes, key=lambda b: (b[0]+b[2])/2)],
                    "annotated_image": str(annotated),
                    "detections": [{"label": result.names[int(b[5])], "confidence": b[4], "xyxy": b[:4]} for b in boxes]}
            report["predictions"].append(item)
            print(json.dumps({k: item[k] for k in ("model", "input", "raw_lines", "left_to_right_labels")}), flush=True)
        report["synthetic_regression"][label] = regression(model)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["synthetic_regression"]), flush=True)


if __name__ == "__main__":
    main()
