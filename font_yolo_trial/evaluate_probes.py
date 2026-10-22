"""Fixed-threshold, held-out qualitative inference; never correct predictions to expected words."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TILE, STRIDE = 384, 288


def nms(boxes, threshold=.5):
    """Greedy class-agnostic NMS for global [x0,y0,x1,y1,confidence,class] boxes."""
    pending = sorted(boxes, key=lambda box: -box[4])
    kept = []
    while pending:
        best, *pending = pending
        kept.append(best)
        remaining = []
        for box in pending:
            intersection = max(0, min(best[2], box[2])-max(best[0], box[0])) * max(
                0, min(best[3], box[3])-max(best[1], box[1]))
            union = (best[2]-best[0])*(best[3]-best[1]) + (box[2]-box[0])*(box[3]-box[1]) - intersection
            if intersection / max(union, 1e-9) <= threshold:
                remaining.append(box)
        pending = remaining
    return kept


def pdf_rect(box, crop, scale):
    return [float(box[i] / scale + crop[i % 2]) for i in range(4)]


def selfcheck():
    boxes = [[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .9, 26], [20, 0, 30, 10, .7, 1]]
    assert [box[5] for box in nms(boxes)] == [26, 1]
    assert nms([]) == []
    assert pdf_rect([4, 8, 20, 24], [10, 20, 30, 40], 4) == [11, 22, 15, 26]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "mixed/training/weights/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "mixed/real_probe_results")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--head", choices=("auto", "end2end", "nms"), default="auto")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    selfcheck()
    if args.self_check:
        print("NMS and PDF-coordinate self-check passed; no model imported or loaded.")
        return
    if not 0 < args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("Confidence and IoU must be in (0, 1]")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from train import read_rows
    from prepare import write_image

    probes = ROOT / "real_probes"
    manifest = json.loads((probes / "manifest.json").read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    head_options = {} if args.head == "auto" else {"end2end": args.head == "end2end"}
    report = {"scope": "Qualitative regression on three previously inspected crops excluded from training; not independent accuracy.",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "weights": str(args.weights.resolve()),
              "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "device": args.device, "confidence_threshold": args.conf, "tile_iou": args.iou,
              "inference_head": args.head,
              "global_class_agnostic_nms_iou": args.iou, "tile_size": TILE, "tile_stride": STRIDE,
              "render_dpi": 288, "resizing": "None; edge tiles padded white", "word_correction": False,
              "tile_ownership": "Centers owned by tile cores with 48px overlap margins",
              "manifest": str(probes / "manifest.json"), "names": model.names, "probes": []}
    for probe in manifest["probes"]:
        path = probes / probe["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == probe["image_sha256"], path
        source = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = source.shape[:2]
        assert [width, height] == probe["image_size_pixels"] and probe["render_dpi"] == 288
        tiles, origins = [], []
        for y in range(0, height, STRIDE):
            for x in range(0, width, STRIDE):
                crop = source[y:y+TILE, x:x+TILE]
                tile = np.full((TILE, TILE, 3), 255, dtype=np.uint8)
                tile[:crop.shape[0], :crop.shape[1]] = crop
                tiles.append(tile)
                origins.append((x, y, crop.shape[1], crop.shape[0]))
        predictions = model.predict(tiles, imgsz=TILE, conf=args.conf, iou=args.iou,
                                    agnostic_nms=True, device=args.device, batch=8, verbose=False,
                                    **head_options)
        boxes = []
        for prediction, (x, y, valid_width, valid_height) in zip(predictions, origins):
            for box in prediction.boxes.data.cpu().tolist():
                # Keep each glyph in the tile where its center is away from a cut edge.
                margin = (TILE - STRIDE) / 2
                left, top = (margin if x else 0), (margin if y else 0)
                right = TILE-margin if x+TILE < width else valid_width
                bottom = TILE-margin if y+TILE < height else valid_height
                if not (left <= (box[0]+box[2])/2 < right and top <= (box[1]+box[3])/2 < bottom):
                    continue
                box[:4] = [max(x, box[0]+x), max(y, box[1]+y),
                           min(x+valid_width, box[2]+x), min(y+valid_height, box[3]+y)]
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
        merged = np.asarray(nms(boxes, args.iou), dtype=np.float32).reshape(-1, 6)
        result = Results(orig_img=source, path=str(path), names=model.names, boxes=merged)
        write_image(output / path.name, result.plot(line_width=1, font_size=10, conf=True))
        report["probes"].append({"id": probe["id"], "source_image": str(path),
            "source_image_sha256": probe["image_sha256"], "source_pdf": probe["source_path"],
            "source_pdf_sha256": probe["source_sha256"], "page_index": probe["page_index"],
            "crop_pdf_rect": probe["crop_pdf_rect"], "annotated_image": str(output / path.name),
            "raw_predicted_lines": read_rows(result), "expected_english_portions": probe["expected_english_portions"],
            "detections": [{"character": model.names[int(box[5])], "class_id": int(box[5]),
                "confidence": float(box[4]), "crop_pixel_xyxy": box[:4].tolist(),
                "pdf_xyxy": pdf_rect(box, probe["crop_pdf_rect"], probe["render_scale_pixels_per_pdf_point"])} for box in merged]})
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "probe_count": len(report["probes"])}, ensure_ascii=True))


if __name__ == "__main__":
    main()
