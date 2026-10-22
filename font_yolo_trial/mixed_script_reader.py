"""Combine preserved YOLO glyphs with automatically cropped local Korean OCR."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics.engine.results import Results

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models_local import _offline
from paddle_ocr_local import model_dirs
from preserved_reader import ROOT, PreservedReader, foreground
from prepare import read_image, write_image
from train import read_rows


def glyph_rows(boxes):
    rows = []
    for box in sorted((b for b in boxes if int(b[5]) < 62), key=lambda b: (b[1]+b[3])/2):
        center, height = (box[1]+box[3])/2, box[3]-box[1]
        row = min(rows, key=lambda r: abs(r["center"]-center), default=None)
        if row is None or abs(row["center"]-center) > height * .6:
            rows.append({"center": center, "boxes": [box]})
        else:
            row["boxes"].append(box)
            row["center"] = float(np.median([(b[1]+b[3])/2 for b in row["boxes"]]))
    result = [{"center": row["center"], "top": float(np.median([b[1] for b in row["boxes"]])),
               "bottom": float(np.median([b[3] for b in row["boxes"]]))} for row in rows]
    return sorted(result, key=lambda r: r["center"])


def expand_detection(box, rows, width, height):
    x0, y0, x1, y1 = box
    center = (y0+y1)/2
    index = min(range(len(rows)), key=lambda i: abs(rows[i]["center"]-center)) if rows else None
    if index is None:
        cap = y1-y0
    else:
        row = rows[index]
        cap = row["bottom"] - row["top"]
        y0, y1 = min(y0, row["top"]), max(y1, row["bottom"])
    pad = max(3, cap * .15)
    x0, y0, x1, y1 = x0-pad, y0-pad, x1+pad, y1+pad
    if index is not None:
        if index > 0:
            y0 = max(y0, (rows[index-1]["bottom"] + rows[index]["top"])/2)
        if index+1 < len(rows):
            y1 = min(y1, (rows[index]["bottom"] + rows[index+1]["top"])/2)
    return {"xyxy": [max(0, int(np.floor(x0))), max(0, int(np.floor(y0))),
                      min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))],
            "row": index, "cap_height": cap}


def recognition_regions(polygons, rows, image):
    height, width = image.shape[:2]
    proposals = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        proposals.append(expand_detection([*points.min(0), *points.max(0)], rows, width, height))
    joined = []
    for item in sorted(proposals, key=lambda p: (-1 if p["row"] is None else p["row"], p["xyxy"][0])):
        if joined and joined[-1]["row"] == item["row"] and item["xyxy"][0] <= joined[-1]["xyxy"][2]:
            a, b = joined[-1]["xyxy"], item["xyxy"]
            joined[-1]["xyxy"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        else:
            joined.append(item)
    ink = foreground(image)
    regions = []
    for item in joined:
        x0, y0, x1, y1 = item["xyxy"]
        if x1 <= x0 or y1 <= y0:
            continue
        patch = ink[y0:y1, x0:x1]
        columns = patch.any(0)
        spans = np.flatnonzero(np.diff(np.r_[False, columns, False].astype(np.int8))).reshape(-1, 2)
        groups = []
        gap = max(2, round(item["cap_height"] * .3))
        for left, right in spans:
            if groups and left-groups[-1][1] <= gap:
                groups[-1][1] = int(right)
            else:
                groups.append([int(left), int(right)])
        for left, right in groups:
            ys, _ = np.where(patch[:, left:right] > 0)
            if not len(ys):
                continue
            regions.append({"xyxy": [x0+left, y0+int(ys.min()), x0+right, y0+int(ys.max())+1],
                            "cap_height": item["cap_height"], "proposal": item["xyxy"]})
    return regions


def korean_body(text, score):
    body = re.sub(r"^[\s:/'\".\-]+|[\s:/'\".\-]+$", "", text)
    return body if score >= .7 and re.fullmatch(r"[가-힣ㄱ-ㅎㅏ-ㅣ\s]+", body) else None


def merge_korean(boxes, names, recognized):
    accepted = [r for r in recognized if r.get("korean_text")]
    kept, suppressed = [], []
    for box in boxes:
        center_x, center_y = (box[0]+box[2])/2, (box[1]+box[3])/2
        region = next((r for r in accepted if r["xyxy"][0] <= center_x <= r["xyxy"][2]
                       and r["xyxy"][1] <= center_y <= r["xyxy"][3]), None)
        if int(box[5]) < 62 and region:
            suppressed.append({"label": names[int(box[5])], "xyxy": box[:4], "replaced_by": region["korean_text"]})
        else:
            kept.append(box)
    combined_names = dict(names)
    for item in accepted:
        index = max(combined_names) + 1
        combined_names[index] = item["korean_text"]
        kept.append([*item["xyxy"], item["score"], index])
    return np.asarray(kept, dtype=np.float32).reshape(-1, 6), combined_names, suppressed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    source = read_image(args.input)
    reader = PreservedReader()
    glyphs, _, details = reader.predict(source, args.input)
    boxes = glyphs.boxes.cpu().numpy().data.tolist()
    rows = glyph_rows(boxes)
    clean = details["cleaned"]
    write_image(args.output / "ocr_input.png", clean)
    det_dir, rec_dir = model_dirs()
    print("Loading local Korean OCR...", flush=True)
    with _offline(paddle_cache=ROOT.parent / ".model_cache/paddlex"):
        from paddleocr import PaddleOCR, TextRecognition
        pipeline = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_detection_model_dir=str(det_dir),
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec", text_recognition_model_dir=str(rec_dir),
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            device="cpu", enable_mkldnn=False)
        result = list(pipeline.predict(input=clean, text_rec_score_thresh=0.0))[0]
        payload = result.json.get("res", result.json)
        polygons = payload.get("dt_polys", payload.get("rec_polys", []))
        regions = recognition_regions(polygons, rows, clean)
        raw_auto = [{"text": str(t), "score": float(s), "polygon": np.asarray(p).tolist()}
                    for t, s, p in zip(payload["rec_texts"], payload["rec_scores"], payload["rec_polys"])]
        crops = []
        for i, item in enumerate(regions):
            x0, y0, x1, y1 = item["xyxy"]
            pad = max(4, round(item["cap_height"] * .25))
            crop = cv2.copyMakeBorder(clean[y0:y1, x0:x1], pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=tuple(int(v) for v in np.median(clean.reshape(-1, 3), axis=0)))
            crops.append(crop)
            write_image(args.output / f"auto_region_{i:02d}.png", crop)
        del pipeline
        recognizer = TextRecognition(model_name="korean_PP-OCRv5_mobile_rec", model_dir=str(rec_dir),
                                     device="cpu", enable_mkldnn=False)
        predictions = list(recognizer.predict(input=crops, batch_size=8)) if crops else []
        assert len(predictions) == len(regions)
        for item, prediction in zip(regions, predictions):
            raw = prediction.json.get("res", prediction.json)
            item.update(text=str(raw.get("rec_text", "")), score=float(raw.get("rec_score", 0)))
            item["korean_text"] = korean_body(item["text"], item["score"])
            print(json.dumps(item, ensure_ascii=True), flush=True)
    merged, names, suppressed = merge_korean(boxes, reader.names, regions)
    combined = Results(orig_img=source, path=str(args.input), names=names, boxes=merged)
    report = {"source": str(args.input.resolve()), "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "manual_crop": False, "word_correction": False, "training": False,
              "yolo_conf": .1, "ocr_acceptance_score": .7, "glyph_rows": rows,
              "yolo_only_lines": read_rows(glyphs), "raw_ocr_auto": raw_auto, "automatic_regions": regions,
              "suppressed_yolo_inside_korean": suppressed, "combined_lines": read_rows(combined),
              "models": {"letters": str(reader.paths["letters"]), "symbols": str(reader.paths["symbols"]),
                         "korean_detection": str(det_dir), "korean_recognition": str(rec_dir)}}
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "recognized.txt").write_text("\n".join(report["combined_lines"]) + "\n", encoding="utf-8")
    panel = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
    for box in merged:
        x0, y0, x1, y1, score, index = box.tolist()
        label = names[int(index)]
        color = (210, 80, 0) if int(index) >= 67 else ((0, 130, 50) if int(index) >= 62 else (20, 60, 200))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.text((int(x0), max(0, int(y0)-19)), f"{label} {score:.2f}", font=font,
                  fill=color, stroke_width=1, stroke_fill="white")
    panel.save(args.output / "combined_detected.png")
    print(json.dumps({"combined_lines": report["combined_lines"]}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
