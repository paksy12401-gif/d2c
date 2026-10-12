"""Local, ground-truth-free OCR audit for dense engineering drawings.

The same PDF page is rendered at two resolutions.  PaddleOCR results are
matched by page position to measure repeatability without sending the drawing
or OCR strings to any external service.  Cross-resolution consistency is a
stability signal, not an accuracy score.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models_local import _offline
from paddle_ocr_local import model_dirs, read_text
from revision_compare import render_document


def _normalise(text: str) -> str:
    return re.sub(r"[^0-9A-Z가-힣ΦØ]+", "", text.upper())


def _centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _match(low: list[dict], high: list[dict], high_scale: float) -> list[dict]:
    candidates = []
    for low_index, left in enumerate(low):
        lx, ly = _centre(left["bbox"])
        lw, lh = left["bbox"][2:]
        for high_index, right in enumerate(high):
            scaled = [value / high_scale for value in right["bbox"]]
            rx, ry = _centre(scaled)
            rw, rh = scaled[2:]
            distance = float(np.hypot(lx - rx, ly - ry))
            tolerance = max(18.0, 0.8 * max(lw, lh, rw, rh))
            if distance <= tolerance:
                candidates.append((distance / tolerance, low_index, high_index))
    used_low, used_high, matches = set(), set(), []
    for relative_distance, low_index, high_index in sorted(candidates):
        if low_index in used_low or high_index in used_high:
            continue
        used_low.add(low_index)
        used_high.add(high_index)
        left, right = low[low_index], high[high_index]
        a, b = _normalise(left["text"]), _normalise(right["text"])
        similarity = SequenceMatcher(None, a, b).ratio() if a or b else 1.0
        matches.append({
            "low_index": low_index,
            "high_index": high_index,
            "low_text": left["text"],
            "high_text": right["text"],
            "low_confidence": left["confidence"],
            "high_confidence": right["confidence"],
            "text_similarity": similarity,
            "exact_normalized_text": a == b,
            "relative_position_distance": relative_distance,
        })
    return matches


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(aw * ah + bw * bh - intersection, 1.0)


def read_text_tiled(gray: np.ndarray, reader, *, min_confidence=.2,
                    tile_size=1200, overlap=120) -> tuple[list[dict], int]:
    """OCR a large sheet in overlapping tiles and merge overlap duplicates."""
    height, width = gray.shape
    found = []
    tile_count = 0
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile_count += 1
            tile = gray[y:min(y + tile_size, height), x:min(x + tile_size, width)]
            # Dense wires and table borders make OCR detection explode in both
            # runtime and false positives. Remove only long axis-aligned runs;
            # short glyph strokes remain available to the recognizer.
            binary = cv2.threshold(tile, 210, 255, cv2.THRESH_BINARY_INV)[1]
            run = max(36, round(min(tile.shape) * .035))
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1)))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
            clean = tile.copy()
            clean[cv2.bitwise_or(horizontal, vertical) > 0] = 255
            for item in read_text(clean, reader, min_confidence=min_confidence):
                item = dict(item)
                item["bbox"] = [item["bbox"][0] + x, item["bbox"][1] + y,
                                item["bbox"][2], item["bbox"][3]]
                item["polygon"] = [[point[0] + x, point[1] + y] for point in item["polygon"]]
                found.append(item)
    kept = []
    for candidate in sorted(found, key=lambda item: item["confidence"], reverse=True):
        cx, cy = _centre(candidate["bbox"])
        duplicate = False
        for other in kept:
            ox, oy = _centre(other["bbox"])
            position_close = np.hypot(cx - ox, cy - oy) <= max(8.0, .35 * min(
                candidate["bbox"][2] + candidate["bbox"][3],
                other["bbox"][2] + other["bbox"][3]))
            text_similar = SequenceMatcher(None, _normalise(candidate["text"]),
                                            _normalise(other["text"])).ratio() >= .5
            if _iou(candidate["bbox"], other["bbox"]) >= .55 or (position_close and text_similar):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, item in enumerate(kept, 1):
        item["id"] = f"TEXT_{index:04d}"
    return kept, tile_count


def propose_text_regions(gray: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    """Find compact horizontal glyph groups after suppressing CAD wires."""
    binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    scale = max(.75, gray.shape[1] / 2150.0)
    run = max(34, round(40 * scale))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
    glyphs = cv2.subtract(binary, cv2.bitwise_or(horizontal, vertical))
    joined = cv2.dilate(glyphs, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, round(5 * scale)), max(1, round(2 * scale)))))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    boxes = []
    for x, y, width, height, area in stats[1:]:
        density = cv2.countNonZero(glyphs[y:y + height, x:x + width]) / max(width * height, 1)
        if (12 * scale <= width <= 420 * scale and 6 * scale <= height <= 48 * scale
                and area >= 10 * scale * scale and .025 <= density <= .85
                and 1.8 <= width / max(height, 1) <= 35):
            pad = max(2, round(3 * scale))
            left, top = max(0, int(x) - pad), max(0, int(y) - pad)
            right = min(gray.shape[1], int(x + width) + pad)
            bottom = min(gray.shape[0], int(y + height) + pad)
            boxes.append([left, top, right - left, bottom - top])
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes, glyphs


def load_recognizer():
    """Load only the local Korean PP-OCRv5 recognizer."""
    _, rec_dir = model_dirs()
    with _offline(paddle_cache=rec_dir.parent.parent):
        from paddleocr import TextRecognition
        model = TextRecognition(model_name="korean_PP-OCRv5_mobile_rec",
                                model_dir=str(rec_dir), device="cpu", enable_mkldnn=False)
    model._cad_model_root = str(rec_dir.parent.parent)
    return model


def recognize_regions(gray: np.ndarray, recognizer, *, min_confidence=.2) -> tuple[list[dict], int]:
    boxes, _ = propose_text_regions(gray)
    crops = []
    valid_boxes = []
    for box in boxes:
        x, y, width, height = box
        crop = gray[y:y + height, x:x + width]
        if crop.size:
            # Upscale tiny CAD lettering before recognition.
            target_height = 48
            factor = max(1.0, target_height / crop.shape[0])
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
            valid_boxes.append(box)
    with _offline(paddle_cache=getattr(recognizer, "_cad_model_root", None)):
        results = recognizer.predict(input=crops, batch_size=64) if crops else []
    texts = []
    for box, result in zip(valid_boxes, results):
        payload = result.json
        payload = payload.get("res", payload)
        text = str(payload.get("rec_text", "")).strip()
        score = float(payload.get("rec_score", 0.0))
        if not text or score < min_confidence:
            continue
        x, y, width, height = box
        texts.append({"id": f"TEXT_{len(texts)+1:04d}", "text": text,
                      "bbox": [float(x), float(y), float(width), float(height)],
                      "confidence": score,
                      "polygon": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                      "links": [], "status": "needs_review"})
    return texts, len(boxes)


def _font(size=18):
    for path in (Path(r"C:\Windows\Fonts\malgun.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _overlay(rgb: np.ndarray, texts: list[dict]) -> Image.Image:
    canvas = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, round(canvas.width / 150)))
    for index, item in enumerate(texts, 1):
        x, y, w, h = item["bbox"]
        confidence = item["confidence"]
        color = (0, 160, 60) if confidence >= .8 else ((230, 150, 0) if confidence >= .5 else (210, 30, 30))
        draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
        draw.text((x, max(0, y - font.size - 2)), str(index), fill=color, font=font,
                  stroke_width=2, stroke_fill="white")
    return canvas


def _review_sheet(rgb: np.ndarray, texts: list[dict], limit=60, *, highest=False) -> Image.Image:
    chosen = sorted(enumerate(texts, 1), key=lambda row: row[1]["confidence"],
                    reverse=highest)[:limit]
    width, row_height = 1500, 92
    sheet = Image.new("RGB", (width, max(row_height, len(chosen) * row_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(20)
    for row, (index, item) in enumerate(chosen):
        x, y, w, h = [int(round(value)) for value in item["bbox"]]
        pad = max(6, h // 2)
        crop = Image.fromarray(rgb).crop((max(0, x - pad), max(0, y - pad),
                                          min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)))
        crop.thumbnail((620, row_height - 10))
        top = row * row_height
        sheet.paste(crop, (10, top + (row_height - crop.height) // 2))
        label = f"#{index:03d}  {item['confidence']:.1%}  {item['text']}"
        draw.text((650, top + 28), label, fill="black", font=font)
        draw.line((0, top + row_height - 1, width, top + row_height - 1), fill=(220, 220, 220))
    return sheet


def audit_document(pdf_path, output_dir, reader=None, *, dpis=(150,), min_confidence=.2):
    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = reader or load_recognizer()
    runs = {}
    images = {}
    data = pdf_path.read_bytes()
    for dpi in dpis:
        rgb = render_document(data, "pdf", dpi=dpi)
        images[dpi] = rgb
        started = time.perf_counter()
        texts, proposal_count = recognize_regions(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), reader,
                                                  min_confidence=min_confidence)
        elapsed = time.perf_counter() - started
        confidences = [item["confidence"] for item in texts]
        runs[dpi] = {
            "dpi": dpi,
            "image_size": [rgb.shape[1], rgb.shape[0]],
            "seconds": elapsed,
            "proposal_count": proposal_count,
            "count": len(texts),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "median_confidence": float(np.median(confidences)) if confidences else 0.0,
            "high_confidence_count": sum(score >= .8 for score in confidences),
            "texts": texts,
        }
    high_dpi = dpis[-1]
    if len(dpis) >= 2:
        low_dpi = dpis[0]
        matches = _match(runs[low_dpi]["texts"], runs[high_dpi]["texts"], high_dpi / low_dpi)
        exact = sum(item["exact_normalized_text"] for item in matches)
        similar = sum(item["text_similarity"] >= .8 for item in matches)
        denominator = max(runs[low_dpi]["count"], runs[high_dpi]["count"], 1)
        consistency = {
            "position_matched_count": len(matches),
            "position_repeatability": len(matches) / denominator,
            "exact_text_count": exact,
            "exact_text_rate_among_matches": exact / max(len(matches), 1),
            "similar_text_count": similar,
            "similar_text_rate_among_matches": similar / max(len(matches), 1),
            "mean_text_similarity": float(np.mean([item["text_similarity"] for item in matches])) if matches else 0.0,
            "note": "Cross-DPI stability only; no ground-truth accuracy/recall is claimed.",
        }
    else:
        matches = []
        consistency = None
    high_texts = runs[high_dpi]["texts"]
    _overlay(images[high_dpi], high_texts).save(output_dir / "overlay.png")
    _review_sheet(images[high_dpi], high_texts).save(output_dir / "low_confidence_review.png")
    _review_sheet(images[high_dpi], high_texts, highest=True).save(
        output_dir / "high_confidence_review.png")
    report = {
        "source_name": pdf_path.name,
        "engine": "OpenCV line suppression/proposals + korean_PP-OCRv5_mobile_rec",
        "api_calls": 0,
        "minimum_confidence": min_confidence,
        "runs": {str(key): value for key, value in runs.items()},
        "cross_dpi_consistency": consistency,
        "matches": matches,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("output")
    args = parser.parse_args()
    result = audit_document(args.pdf, args.output)
    print(json.dumps({"source": result["source_name"],
                      "runs": {k: {x: v[x] for x in ("count", "mean_confidence", "seconds")}
                               for k, v in result["runs"].items()},
                      "consistency": result["cross_dpi_consistency"]}, ensure_ascii=False, indent=2))
