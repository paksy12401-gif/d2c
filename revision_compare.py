"""Local visual revision comparison for drawing PDFs and raster images.

The module never calls a network service.  It aligns the AFTER page to the
BEFORE page, finds red revision annotations when present, and falls back to a
tolerant raster-ink difference for unannotated drawings.  PaddleOCR can be
passed in to read only a compact sheet made from red annotation glyphs.
"""
from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import pypdfium2 as pdfium


MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000


def render_document(data: bytes, file_type: str, *, dpi: int = 100) -> np.ndarray:
    """Render the first page as RGB uint8 at a bounded resolution."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("비어 있지 않은 20MB 이하 파일이 필요합니다.")
    if type(dpi) is not int or not 72 <= dpi <= 160:
        raise ValueError("비교 렌더링 DPI는 72~160 정수여야 합니다.")
    kind = str(file_type).lower().lstrip(".")
    if kind in ("pdf", "application/pdf"):
        try:
            with pdfium.PdfDocument(data) as document:
                if not len(document):
                    raise ValueError("페이지가 없는 PDF입니다.")
                page = document[0]
                width, height = page.get_size()
                scale = dpi / 72.0
                if math.ceil(width * scale) * math.ceil(height * scale) > MAX_PIXELS:
                    raise ValueError("렌더링 결과가 2,000만 픽셀을 초과합니다. DPI를 낮추세요.")
                bitmap = page.render(scale=scale)
                try:
                    return np.asarray(bitmap.to_pil().convert("RGB"), dtype=np.uint8).copy()
                finally:
                    bitmap.close()
        except pdfium.PdfiumError as exc:
            raise ValueError("PDF를 읽을 수 없습니다.") from exc
    if kind in ("png", "image/png", "jpg", "jpeg", "image/jpeg"):
        try:
            with Image.open(BytesIO(data)) as source:
                width, height = source.size
                if width * height > MAX_PIXELS:
                    raise ValueError("이미지가 2,000만 픽셀을 초과합니다.")
                return np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("PNG/JPEG 이미지를 읽을 수 없습니다.") from exc
    raise ValueError("PDF, PNG 또는 JPEG 파일이 필요합니다.")


def _gray(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB uint8 이미지가 필요합니다.")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def align_pages(before_rgb: np.ndarray, after_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Align AFTER to BEFORE with local ORB features and a robust affine fit."""
    before_gray, after_gray = _gray(before_rgb), _gray(after_rgb)
    bh, bw = before_gray.shape
    ah, aw = after_gray.shape
    if abs((bw / bh) - (aw / ah)) > 0.03:
        raise ValueError("두 페이지의 가로세로 비율 차이가 너무 큽니다.")
    resized_after = cv2.resize(after_gray, (bw, bh), interpolation=cv2.INTER_AREA)
    analysis_scale = min(1.0, 1800.0 / bw)
    size = (max(1, round(bw * analysis_scale)), max(1, round(bh * analysis_scale)))
    left = cv2.resize(before_gray, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(resized_after, size, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=12000, fastThreshold=10)
    key_left, desc_left = orb.detectAndCompute(left, None)
    key_right, desc_right = orb.detectAndCompute(right, None)
    if desc_left is None or desc_right is None:
        raise ValueError("도면 정렬용 특징점을 찾지 못했습니다.")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_right, desc_left, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 12:
        raise ValueError("두 도면을 안정적으로 정렬할 공통 특징점이 부족합니다.")
    source = np.float32([key_right[item.queryIdx].pt for item in good])
    target = np.float32([key_left[item.trainIdx].pt for item in good])
    small_matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=5000, confidence=0.995,
    )
    if small_matrix is None or inliers is None:
        raise ValueError("도면 정렬 행렬을 계산하지 못했습니다.")
    ratio = float(inliers.mean())
    linear = small_matrix[:, :2]
    scale_value = math.sqrt(abs(float(np.linalg.det(linear))))
    rotation = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    if ratio < 0.45 or not 0.94 <= scale_value <= 1.06 or abs(rotation) > 3.0:
        raise ValueError("도면 정렬 신뢰도가 낮습니다. 동일한 시트/축척인지 확인하세요.")
    fit = small_matrix.copy()
    fit[:, 2] /= analysis_scale
    resize_matrix = np.array([[bw / aw, 0, 0], [0, bh / ah, 0], [0, 0, 1]], dtype=float)
    fit_h = np.vstack((fit, [0, 0, 1])) @ resize_matrix
    matrix = fit_h[:2]
    # Nearest-neighbour keeps one-pixel CAD strokes and coloured revision boxes
    # intact. Bilinear interpolation fragments those strokes into OCR-like blobs.
    aligned = cv2.warpAffine(after_rgb, matrix, (bw, bh), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return aligned, {
        "method": "ORB + RANSAC affine",
        "matrix_after_to_before": matrix.tolist(),
        "matched_features": len(good),
        "inlier_ratio": ratio,
        "estimated_rotation_deg": rotation,
        "estimated_scale": scale_value,
    }


def red_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 70), (15, 255, 255)),
        cv2.inRange(hsv, (165, 70, 70), (180, 255, 255)),
    )


def _union(boxes):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def _gap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0)
    dy = max(by - (ay + ah), ay - (by + bh), 0)
    return math.hypot(dx, dy)


def _red_text_lines(mask: np.ndarray) -> list[list[int]]:
    factor = max(0.75, mask.shape[1] / 3118.0)
    long_side = max(18, round(24 * factor))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (long_side, 1)))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, long_side)))
    glyphs = cv2.subtract(mask, cv2.bitwise_or(horizontal, vertical))
    joined = cv2.dilate(glyphs, cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(5, round(9 * factor)), max(1, round(3 * factor)))))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    boxes = []
    for x, y, width, height, area in stats[1:]:
        if area >= 8 * factor * factor and width >= 7 * factor and 2 * factor <= height <= 42 * factor:
            boxes.append([int(x), int(y), int(width), int(height)])
    # Merge syllable/word fragments on the same line.
    changed = True
    while changed:
        changed, output = False, []
        while boxes:
            first = boxes.pop(0)
            ax, ay, aw, ah = first
            merged = False
            for index, second in enumerate(boxes):
                bx, by, bw, bh = second
                overlap = min(ay + ah, by + bh) - max(ay, by)
                horizontal_gap = max(bx - (ax + aw), ax - (bx + bw), 0)
                if overlap >= min(ah, bh) * 0.4 and horizontal_gap <= 22 * factor:
                    boxes[index] = _union([first, second])
                    changed, merged = True, True
                    break
            if not merged:
                output.append(first)
        boxes = output
    # Short remnants are normally corners/intersections from revision boxes,
    # not readable annotation text.
    boxes = [box for box in boxes if box[2] >= 25 * factor and box[3] >= 10 * factor]
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def _pack_text_lines(mask: np.ndarray, boxes: list[list[int]]) -> tuple[np.ndarray, list[dict]]:
    if not boxes:
        return np.full((32, 256), 255, np.uint8), []
    scale, padding = 2, 12
    width = max(800, max(box[2] for box in boxes) * scale + 2 * padding)
    heights = [box[3] * scale + 2 * padding for box in boxes]
    sheet = np.full((sum(heights), width), 255, np.uint8)
    mapping, cursor = [], 0
    for index, (x, y, w, h) in enumerate(boxes):
        crop = np.full((h, w), 255, np.uint8)
        crop[mask[y:y + h, x:x + w] > 0] = 0
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        packed = [padding, cursor + padding, crop.shape[1], crop.shape[0]]
        sheet[packed[1]:packed[1] + packed[3], packed[0]:packed[0] + packed[2]] = crop
        mapping.append({"index": index, "source": [x, y, w, h], "packed": packed})
        cursor += crop.shape[0] + 2 * padding
    return sheet, mapping


def _annotation_text(mask: np.ndarray, ocr_reader, min_confidence: float) -> list[dict]:
    boxes = _red_text_lines(mask)
    if not boxes:
        return []
    sheet, mapping = _pack_text_lines(mask, boxes)
    by_index = {item["index"]: [] for item in mapping}
    if ocr_reader is not None:
        from models_local import recognize_texts
        for text in recognize_texts(sheet, ocr_reader, min_confidence=min_confidence):
            _, cy, _, _ = text["bbox"]
            cy += text["bbox"][3] / 2
            for item in mapping:
                _, py, _, ph = item["packed"]
                if py - 8 <= cy <= py + ph + 8:
                    by_index[item["index"]].append(text)
                    break
    lines = []
    for item in mapping:
        found = by_index[item["index"]]
        label = " / ".join(part["text"].strip() for part in found if part["text"].strip())
        confidence = float(np.mean([part["confidence"] for part in found])) if found else None
        lines.append({"bbox": item["source"], "text": label, "confidence": confidence})
    return lines


def _normalize_annotation(text: str) -> str:
    """Apply narrow, reviewable corrections for common Korean OCR variants."""
    replacements = {
        "색션": "섹션", "석션": "섹션", "]PRT": "PRT",
        "B/CABLE변경": "B/CABLE 변경", "EXTN분기": "EXTN 분기",
        "변경,치수": "변경, 치수", "RH1_6분기": "RH1_6 분기",
        "PRT변경": "PRT 변경", "내부치수": "내부 치수", " / 삭제": " 삭제",
    }
    normalized = text.strip()
    for before, after in replacements.items():
        normalized = normalized.replace(before, after)
    return normalized


def _group_lines(lines: list[dict], width: int) -> list[dict]:
    if not lines:
        return []
    factor = max(0.75, width / 3118.0)
    parent = list(range(len(lines)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i]["bbox"], lines[j]["bbox"]
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            vertical_gap = max(by - (ay + ah), ay - (by + bh), 0)
            overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
            overlap_y = min(ay + ah, by + bh) - max(ay, by)
            horizontal_gap = max(bx - (ax + aw), ax - (bx + bw), 0)
            same_row = overlap_y > 0 and horizontal_gap <= 45 * factor
            next_row = vertical_gap <= 10 * factor and overlap_x > min(aw, bw) * 0.25
            if same_row or next_row:
                parent[root(j)] = root(i)
    groups = {}
    for index, line in enumerate(lines):
        groups.setdefault(root(index), []).append(line)
    result = []
    for values in groups.values():
        values.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        usable = [item for item in values if item["text"] and not (
            len(item["text"].strip()) <= 2 and (item["confidence"] or 0) < 0.8)]
        label = _normalize_annotation(" / ".join(item["text"] for item in usable))
        scores = [item["confidence"] for item in usable if item["confidence"] is not None]
        result.append({"annotation_bbox": _union([item["bbox"] for item in values]),
                       "label": label, "ocr_confidence": float(np.mean(scores)) if scores else None})
    return sorted(result, key=lambda item: (item["annotation_bbox"][1], item["annotation_bbox"][0]))


def _red_rectangles(mask: np.ndarray) -> list[list[int]]:
    factor = max(0.75, mask.shape[1] / 3118.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if w >= 20 * factor and h >= 12 * factor and w * h >= 450 * factor * factor and len(approx) <= 10:
            candidates.append([x, y, w, h])
    candidates.sort(key=lambda box: box[2] * box[3], reverse=True)
    kept = []
    for box in candidates:
        x, y, w, h = box
        if any(abs(x - k[0]) <= 4 * factor and abs(y - k[1]) <= 4 * factor
               and abs(x + w - (k[0] + k[2])) <= 6 * factor
               and abs(y + h - (k[1] + k[3])) <= 6 * factor for k in kept):
            continue
        kept.append(box)
    return kept


def _ink_changes(before_rgb: np.ndarray, after_rgb: np.ndarray, ignore_mask=None,
                 *, ink_threshold=205, tolerance_px=2.0):
    old = _gray(before_rgb) < ink_threshold
    new = _gray(after_rgb) < ink_threshold
    distance_to_new = cv2.distanceTransform((~new).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_old = cv2.distanceTransform((~old).astype(np.uint8), cv2.DIST_L2, 3)
    removed = old & (distance_to_new > tolerance_px)
    added = new & (distance_to_old > tolerance_px)
    if ignore_mask is not None:
        ignored = cv2.dilate(ignore_mask, np.ones((5, 5), np.uint8)) > 0
        removed[ignored] = False
        added[ignored] = False
    return (removed | added).astype(np.uint8) * 255, added, removed


def _ink_difference(before_rgb: np.ndarray, after_rgb: np.ndarray, ignore_mask=None) -> np.ndarray:
    return _ink_changes(before_rgb, after_rgb, ignore_mask)[0]


def _region_label(box, shape, added_count, removed_count):
    x, y, width, height = box
    page_height, page_width = shape[:2]
    horizontal = ("좌측", "중앙", "우측")[min(2, int((x + width / 2) / page_width * 3))]
    vertical = ("상단", "중단", "하단")[min(2, int((y + height / 2) / page_height * 3))]
    if added_count > removed_count * 1.5:
        kind = "형상 추가 중심"
    elif removed_count > added_count * 1.5:
        kind = "형상 삭제 중심"
    else:
        kind = "형상 교체·이동"
    return f"{horizontal} {vertical} · {kind} 후보"


def _difference_regions(mask: np.ndarray, limit=30, *, join_size=(17, 11),
                        join_iterations=2, min_area=180, min_width=12, min_height=8) -> list[list[int]]:
    joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, join_size),
                        iterations=join_iterations)
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    boxes = [[int(x), int(y), int(w), int(h)] for x, y, w, h, area in stats[1:]
             if area >= min_area and w >= min_width and h >= min_height]
    return sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)[:limit]


def _expand(box, shape, padding):
    x, y, w, h = box
    height, width = shape[:2]
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(width, x + w + padding), min(height, y + h + padding)
    return [x1, y1, x2 - x1, y2 - y1]


def _crop(rgb, box):
    x, y, w, h = box
    return rgb[y:y + h, x:x + w].copy()


def compare_documents(before_data: bytes, after_data: bytes, before_type: str, after_type: str,
                      *, dpi: int = 100, ocr_reader=None, min_confidence: float = 0.2,
                      sensitivity: str = "balanced") -> dict:
    """Return aligned pages, overview, crops, and a serializable change report."""
    before = render_document(before_data, before_type, dpi=dpi)
    after = render_document(after_data, after_type, dpi=dpi)
    aligned, alignment = align_pages(before, after)
    revisions = red_mask(aligned)
    red_pixels = int(np.count_nonzero(revisions))
    sensitivity_settings = {
        "high": {"ink_threshold": 225, "tolerance_px": .75, "limit": 100,
                 "join_size": (9, 5), "join_iterations": 1,
                 "min_area": 24, "min_width": 3, "min_height": 3},
        "balanced": {"ink_threshold": 205, "tolerance_px": 2.0, "limit": 30,
                     "join_size": (17, 11), "join_iterations": 2,
                     "min_area": 180, "min_width": 12, "min_height": 8},
        "strict": {"ink_threshold": 190, "tolerance_px": 3.0, "limit": 20,
                   "join_size": (19, 13), "join_iterations": 2,
                   "min_area": 300, "min_width": 16, "min_height": 10},
    }
    if sensitivity not in sensitivity_settings:
        raise ValueError("변경 민감도는 high/balanced/strict 중 하나여야 합니다.")
    settings = sensitivity_settings[sensitivity]
    difference, added_ink, removed_ink = _ink_changes(
        before, aligned, revisions if red_pixels >= 100 else None,
        ink_threshold=settings["ink_threshold"], tolerance_px=settings["tolerance_px"])
    items, difference_stats = [], None
    use_red_annotations = red_pixels >= 100 and sensitivity != "high"
    if use_red_annotations:
        lines = _annotation_text(revisions, ocr_reader, min_confidence)
        annotations = _group_lines(lines, before.shape[1])
        rectangles = _red_rectangles(revisions)
        max_distance = 260 * max(0.75, before.shape[1] / 3118.0)
        for index, annotation in enumerate(annotations, 1):
            annotation_box = annotation["annotation_bbox"]
            nearby = min(rectangles, key=lambda box: _gap(annotation_box, box), default=None)
            target_box = nearby if nearby is not None and _gap(annotation_box, nearby) <= max_distance else None
            combined = _union([annotation_box] + ([target_box] if target_box else []))
            box = _expand(combined, before.shape, round(24 * max(0.75, before.shape[1] / 3118.0)))
            label = annotation["label"] or f"빨간 변경 주석 영역 {index}"
            items.append({"id": f"CHANGE_{index:03d}", "label": label,
                          "ocr_confidence": annotation["ocr_confidence"],
                          "bbox_px": box, "annotation_bbox_px": annotation_box,
                          "target_bbox_px": target_box, "source": "red_annotation"})
    else:
        all_regions = _difference_regions(
            difference, limit=10000, join_size=settings["join_size"],
            join_iterations=settings["join_iterations"], min_area=settings["min_area"],
            min_width=settings["min_width"], min_height=settings["min_height"])
        shown_regions = all_regions[:settings["limit"]]
        covered = np.zeros(difference.shape, dtype=bool)
        for x, y, width, height in shown_regions:
            covered[y:y + height, x:x + width] = True
        changed_pixels = int(np.count_nonzero(difference))
        difference_stats = {
            "all_region_count": len(all_regions), "shown_region_count": len(shown_regions),
            "changed_ink_pixels": changed_pixels,
            "changed_page_ratio": changed_pixels / difference.size,
            "shown_changed_pixel_coverage": (int(np.count_nonzero((difference > 0) & covered)) / changed_pixels
                                               if changed_pixels else 1.0),
            "added_ink_pixels": int(np.count_nonzero(added_ink)),
            "removed_ink_pixels": int(np.count_nonzero(removed_ink)),
            "sensitivity": sensitivity,
            "ink_threshold": settings["ink_threshold"],
            "position_tolerance_px": settings["tolerance_px"],
        }
        for index, box in enumerate(shown_regions, 1):
            x, y, width, height = box
            added_count = int(np.count_nonzero(added_ink[y:y + height, x:x + width]))
            removed_count = int(np.count_nonzero(removed_ink[y:y + height, x:x + width]))
            items.append({"id": f"CHANGE_{index:03d}",
                          "label": _region_label(box, before.shape, added_count, removed_count),
                          "ocr_confidence": None, "bbox_px": _expand(box, before.shape, 24),
                          "annotation_bbox_px": None, "target_bbox_px": box,
                          "added_ink_pixels": added_count, "removed_ink_pixels": removed_count,
                          "source": "raster_difference"})
    overlay = aligned.copy()
    for item in items:
        x, y, w, h = item["bbox_px"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 120, 255), 3)
        cv2.putText(overlay, item["id"], (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 80, 220), 2, cv2.LINE_AA)
        item["before_crop"] = _crop(before, item["bbox_px"])
        item["after_crop"] = _crop(aligned, item["bbox_px"])
    warnings = ([
        "빨간 주석은 변경 후보의 위치·설명이며 실제 설계 변경의 진위는 원본과 함께 검토해야 합니다.",
        "OCR 문자열과 자동 정렬 결과는 확정값이 아닙니다.",
    ] if use_red_annotations else [
        "빨간 정답 없이 정렬 후 2픽셀을 초과한 선 흔적 차이를 묶은 블라인드 후보입니다.",
        "추가·삭제·교체 분류는 픽셀 수 기반 후보이며 부품 의미와 전기 연결을 확정하지 않습니다.",
    ])
    return {"before": before, "after_aligned": aligned, "overview": overlay,
            "difference_mask": difference, "alignment": alignment,
            "red_annotation_pixels": red_pixels, "items": items,
            "difference_stats": difference_stats, "warnings": warnings}


def serializable_report(result: dict) -> dict:
    return {
        "alignment": result["alignment"],
        "red_annotation_pixels": result["red_annotation_pixels"],
        "difference_stats": result.get("difference_stats"),
        "change_count": len(result["items"]),
        "changes": [{key: value for key, value in item.items()
                     if key not in ("before_crop", "after_crop")} for item in result["items"]],
        "warnings": result["warnings"],
    }


def save_result(result: dict, output_dir) -> Path:
    output = Path(output_dir)
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["overview"]).save(output / "overview.png")
    Image.fromarray(result["before"]).save(output / "before.png")
    Image.fromarray(result["after_aligned"]).save(output / "after_aligned.png")
    for item in result["items"]:
        Image.fromarray(item["before_crop"]).save(crops / f"{item['id']}_before.png")
        Image.fromarray(item["after_crop"]).save(crops / f"{item['id']}_after.png")
    report_path = output / "report.json"
    report_path.write_text(json.dumps(serializable_report(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
