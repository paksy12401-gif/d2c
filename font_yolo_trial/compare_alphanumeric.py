"""Compare alphabet and alphanumeric models with identical, fixed input transforms."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path("C:/Users/Win11Pro/AppData/Local/Temp/codex-clipboard-6d0164de-59ad-4b42-8741-1675a80e10ca.png")
SCALE, CANVAS, OFFSET_X, OFFSET_Y = .4, 384, 70, 150
CONF, IOU = .25, .5


def normalize_source(source):
    resized = cv2.resize(source, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    height, width = resized.shape[:2]
    if OFFSET_X + width > CANVAS or OFFSET_Y + height > CANVAS:
        raise ValueError("The fixed diagnostic transform does not fit this image")
    canvas = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
    canvas[OFFSET_Y:OFFSET_Y+height, OFFSET_X:OFFSET_X+width] = resized
    return canvas


def selfcheck():
    source = np.full((137, 549, 3), 255, dtype=np.uint8)
    source[20:40, 40:60] = 0
    before = source.copy()
    normalized = normalize_source(source)
    resized = cv2.resize(source, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    height, width = resized.shape[:2]
    assert normalized.shape == (384, 384, 3)
    assert (width, height) == (220, 55)
    assert np.array_equal(normalized[150:150+height, 70:70+width], resized)
    assert np.all(normalized[:150] == 255) and np.all(normalized[:, :70] == 255)
    assert np.array_equal(source, before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    selfcheck()
    if args.self_check:
        print("Fixed normalization dimensions and pixels checked; no model loaded.")
        return

    from ultralytics import YOLO
    from prepare import write_image
    from train import read_rows

    weights = {"old": ROOT / "mixed/training/weights/best.pt",
               "new": ROOT / "alphanumeric/training/weights/best.pt"}
    for path in [SOURCE, *weights.values()]:
        if not path.is_file():
            raise FileNotFoundError(path)
    source = cv2.imdecode(np.fromfile(SOURCE, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Cannot decode {SOURCE}")
    inputs = {"original": (source, 640), "normalized": (normalize_source(source), 384)}
    output = ROOT / "alphanumeric/user_comparison"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Diagnostic prior-seen user sample; not an independent accuracy test.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_size_pixels": [source.shape[1], source.shape[0]],
        "visually_inferred_text": "bll15914",
        "text_status": "Not user-confirmed; not used to correct or score model predictions.",
        "device": args.device, "confidence_threshold": CONF, "iou": IOU,
        "agnostic_nms": False, "word_correction": False,
        "normalization": {"scale": SCALE, "interpolation": "cv2.INTER_AREA",
                          "canvas_size_pixels": [CANVAS, CANVAS], "background_bgr": [255, 255, 255],
                          "offset_xy": [OFFSET_X, OFFSET_Y]},
        "inputs": {}, "models": {}, "predictions": [],
    }
    for transform, (pixels, image_size) in inputs.items():
        input_path = output / f"input_{transform}.png"
        write_image(input_path, pixels)
        report["inputs"][transform] = {
            "path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "size_pixels": [pixels.shape[1], pixels.shape[0]], "inference_imgsz": image_size,
        }
    for label, path in weights.items():
        model = YOLO(str(path))
        report["models"][label] = {
            "weights": str(path), "weights_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "names": model.names,
        }
        for transform, (pixels, image_size) in inputs.items():
            result = model.predict(pixels.copy(), imgsz=image_size, conf=CONF, iou=IOU,
                                   agnostic_nms=False, device=args.device, verbose=False)[0]
            annotated = output / f"{label}_{transform}.png"
            write_image(annotated, result.plot(line_width=1, font_size=10, labels=True, conf=True))
            report["predictions"].append({
                "model": label, "transform": transform, "annotated_image": str(annotated),
                "raw_predicted_lines": read_rows(result),
                "detections": [{"character": result.names[int(box[5])], "class_id": int(box[5]),
                                "confidence": box[4], "input_pixel_xyxy": box[:4]}
                               for box in result.boxes.data.cpu().tolist()],
            })
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "predictions": [
        {key: item[key] for key in ("model", "transform", "raw_predicted_lines")}
        for item in report["predictions"]]}, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
