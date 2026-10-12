"""Search previously missed symbols in high-resolution PDF tiles, without YOLO.

Run this file or the companion notebook. All detections are review candidates,
not measured precision/recall.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from shape_match_all import IMAGE_ROOT, LABEL_ROOT, PDF_PATH, edge_similarity, load_templates, same_position


WORKSPACE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PRIOR_CSV = WORKSPACE / "shape_match_all_output" / "20261012_174753_645724" / "all_class_candidates.csv"
OUTPUT_ROOT = WORKSPACE / "small_symbol_highres_output"
DPI = 288
SEARCH_RIGHT_PT = 5200  # The drawing occupies the left side; the right side is a parts table.
TILE_W_PT = 800
TILE_H_PT = 900
OVERLAP_PT = 100
RATIO = 0.85
RANSAC_PX = 6
MAX_PER_TEMPLATE_TILE = 8


def missing_classes():
    with PRIOR_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["class_name"]
            for row in csv.DictReader(handle)
            if row["status"] == "not_detected"
        }


def positions(total, tile, overlap):
    result = list(range(0, max(1, math.ceil(total - tile)) + 1, tile - overlap))
    last = max(0, math.ceil(total - tile))
    if result[-1] < last:
        result.append(last)
    return sorted(set(result))


def make_candidate(template, gray, affine, inliers, matches, clip):
    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    if not 0.1 <= scale <= 2:
        return None
    h, w = template["crop"].shape
    corners = cv2.transform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), affine
    ).reshape(-1, 2)
    if not np.isfinite(corners).all():
        return None
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    factor = DPI / 72
    bbox = [
        float(lo[0] / factor + clip.x0),
        float(lo[1] / factor + clip.y0),
        float(hi[0] / factor + clip.x0),
        float(hi[1] / factor + clip.y0),
    ]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if not (8 <= bw <= 250 and 8 <= bh <= 250):
        return None
    if not (
        clip.x0 <= bbox[0] < bbox[2] <= clip.x1
        and clip.y0 <= bbox[1] < bbox[3] <= clip.y1
    ):
        return None
    score = edge_similarity(template["crop"], gray, np.vstack([affine, [0, 0, 1]]))
    if inliers >= 8 and score >= 0.60:
        status = "strong_candidate"
    elif inliers >= 5 and score >= 0.40:
        status = "review"
    else:
        return None
    return {
        "class_name": template["class_name"],
        "status": status,
        "inliers": int(inliers),
        "edge_score": round(score, 4),
        "feature_matches": int(matches),
        "source_image": template["image"],
        "bbox_pdf_pt": [round(v, 2) for v in bbox],
    }


def search_tile(gray, clip, templates, sift):
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) < 2:
        return []
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    matcher.add([descriptors])
    matcher.train()
    found = []
    for template in templates:
        source_desc = template["descriptors"]
        if source_desc is None or len(source_desc) < 5:
            continue
        pairs = matcher.knnMatch(source_desc, k=2)
        remaining = [
            p[0]
            for p in pairs
            if len(p) == 2 and p[0].distance < RATIO * p[1].distance
        ]
        for _ in range(MAX_PER_TEMPLATE_TILE):
            if len(remaining) < 5:
                break
            source = np.float32(
                [template["keypoints"][m.queryIdx].pt for m in remaining]
            ).reshape(-1, 1, 2)
            target = np.float32(
                [keypoints[m.trainIdx].pt for m in remaining]
            ).reshape(-1, 1, 2)
            affine, mask = cv2.estimateAffinePartial2D(
                source, target, method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_PX, maxIters=10000,
            )
            if affine is None or mask is None:
                break
            inliers = int(mask.sum())
            if inliers >= 5:
                candidate = make_candidate(
                    template, gray, affine, inliers, len(remaining), clip
                )
                if candidate:
                    found.append(candidate)
            remaining = [
                m for m, used in zip(remaining, mask.ravel()) if not used
            ]
    return found


def deduplicate(candidates):
    candidates.sort(
        key=lambda c: (
            c["status"] == "strong_candidate", c["edge_score"], c["inliers"]
        ),
        reverse=True,
    )
    results = []
    for candidate in candidates:
        if any(
            other["class_name"] == candidate["class_name"]
            and same_position(candidate, other)
            for other in results
        ):
            continue
        results.append(candidate)
    return results


def save_results(output_dir, classes, detections):
    output_dir.mkdir(parents=True, exist_ok=False)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir()
    by_class = defaultdict(list)
    with pymupdf.open(PDF_PATH) as doc:
        page = doc[0]
        clean = page.get_displaylist(annots=False)
        for n, row in enumerate(detections, 1):
            by_class[row["class_name"]].append(row)
            box = pymupdf.Rect(row["bbox_pdf_pt"])
            clip = pymupdf.Rect(box.x0 - 12, box.y0 - 12, box.x1 + 12, box.y1 + 12) & page.rect
            path = crops_dir / f"{n:03d}_{row['class_name']}.png"
            clean.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=clip, alpha=False).save(path)
            row["crop_file"] = str(path)
            color = (0, 0.6, 0) if row["status"] == "strong_candidate" else (1, .55, 0)
            page.draw_rect(box, color=color, width=1.5)
            page.insert_text((box.x0, max(6, box.y0 - 4)), f"{row['class_name']} {row['edge_score']:.2f}",
                             fontsize=6, color=color)
        doc.save(output_dir / "small_symbol_candidates.pdf", garbage=4, deflate=True)
    summary = []
    for name in sorted(classes):
        values = by_class[name]
        status = "strong_candidate" if any(v["status"] == "strong_candidate" for v in values) else "review" if values else "not_detected"
        summary.append({"class_name": name, "status": status, "candidate_count": len(values)})
    with (output_dir / "small_symbol_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "status", "candidate_count"])
        writer.writeheader(); writer.writerows(summary)
    (output_dir / "small_symbol_candidates.json").write_text(
        json.dumps({"source_pdf": str(PDF_PATH), "dpi": DPI,
                    "method": "high-resolution tiled SIFT plus edge verification, no YOLO",
                    "scope": "previously missing classes, left drawing region only",
                    "summary": summary, "detections": detections}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    columns, cw, ch = 5, 250, 220
    gallery = Image.new("RGB", (columns * cw, max(1, math.ceil(len(detections) / columns)) * ch), "white")
    pen = ImageDraw.Draw(gallery)
    for index, row in enumerate(detections):
        left, top = index % columns * cw, index // columns * ch
        crop = Image.open(row["crop_file"]).convert("RGB")
        crop.thumbnail((cw - 16, ch - 44))
        gallery.paste(crop, (left + (cw - crop.width) // 2, top + 36))
        pen.rectangle((left, top, left + cw - 1, top + ch - 1), outline="green" if row["status"] == "strong_candidate" else "orange", width=2)
        pen.text((left + 5, top + 5), f"{row['class_name']} {row['edge_score']:.2f} {row['status']}", fill="black")
    gallery.save(output_dir / "small_symbol_gallery.png")
    return summary


def main():
    for path in (PDF_PATH, IMAGE_ROOT, LABEL_ROOT, PRIOR_CSV):
        if not path.exists():
            raise FileNotFoundError(path)
    classes = missing_classes()
    sift = cv2.SIFT_create(nfeatures=25000)
    templates = [t for t in load_templates(sift) if t["class_name"] in classes]
    candidates = []
    with pymupdf.open(PDF_PATH) as doc:
        page = doc[0]
        right = min(SEARCH_RIGHT_PT, page.rect.x1)
        x_values = positions(right, TILE_W_PT, OVERLAP_PT)
        y_values = positions(page.rect.y1, TILE_H_PT, OVERLAP_PT)
        total = len(x_values) * len(y_values)
        print(f"고해상도 검색: {DPI} DPI / {len(classes)}클래스 / {len(templates)}템플릿 / {total}타일", flush=True)
        for index, (x0, y0) in enumerate(((x, y) for y in y_values for x in x_values), 1):
            clip = pymupdf.Rect(x0, y0, x0 + TILE_W_PT, y0 + TILE_H_PT) & pymupdf.Rect(0, 0, right, page.rect.y1)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(DPI / 72, DPI / 72),
                                  clip=clip, colorspace=pymupdf.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            tile_candidates = search_tile(gray, clip, templates, sift)
            candidates.extend(tile_candidates)
            print(f"타일 {index}/{total}: 후보 {len(tile_candidates)}개", flush=True)
    detections = deduplicate(candidates)
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary = save_results(output_dir, classes, detections)
    print("결과 폴더:", output_dir)
    print("클래스별 결과:", summary)
    from merge_shape_candidates import main as merge_main
    merge_main(fine_path=output_dir / "small_symbol_candidates.json")
    return output_dir


if __name__ == "__main__":
    main()
