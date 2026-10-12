"""Search all registered CAD symbols in one vector PDF page without YOLO."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json
import math

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw


ROOT = Path(r"C:\Users\Win11Pro\Desktop\drawing")
PDF_PATH = Path(
    r"C:\Users\Win11Pro\Desktop\1. DWG 객체 변환 도면 생성 샘플"
    r"\DWG 객체 변환 샘플.pdf"
)
IMAGE_ROOT = ROOT / "dataset_640"
LABEL_ROOT = ROOT / "labels_640"
OUTPUT_DIR = Path.cwd() / "shape_match_all_output" / datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

PAGE_INDEX = 0
DPI = 72
RATIO_TEST = 0.75
RANSAC_ERROR = 4.0
MAX_INSTANCES_PER_TEMPLATE = 4


def load_templates(sift):
    templates = []
    for path in sorted(IMAGE_ROOT.rglob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        label = LABEL_ROOT / path.parent.name / (path.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(f"정답박스가 없습니다: {label}")
        parts = label.read_text(encoding="utf-8").split()
        if len(parts) != 5:
            raise ValueError(f"정답박스 형식 오류: {label}")
        _, cx, cy, bw, bh = map(float, parts)
        image = np.asarray(Image.open(path).convert("L"))
        height, width = image.shape
        x1, y1, x2, y2 = [
            round(value)
            for value in (
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
        ]
        crop = image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
        if crop.size == 0:
            raise ValueError(f"빈 정답박스: {label}")
        keypoints, descriptors = sift.detectAndCompute(crop, None)
        templates.append(
            {
                "class_name": path.parent.name,
                "image": str(path),
                "crop": crop,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }
        )
    return templates


def edge_similarity(template, page_gray, homography):
    try:
        aligned = cv2.warpPerspective(
            page_gray,
            np.linalg.inv(homography),
            (template.shape[1], template.shape[0]),
            borderValue=255,
        )
    except np.linalg.LinAlgError:
        return 0.0
    reference = cv2.Canny(template, 70, 180)
    observed = cv2.Canny(aligned, 70, 180)
    if not np.any(reference) or not np.any(observed):
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    reference_covered = np.mean(cv2.dilate(observed, kernel)[reference > 0] > 0)
    observed_covered = np.mean(cv2.dilate(reference, kernel)[observed > 0] > 0)
    return float(
        2 * reference_covered * observed_covered
        / max(reference_covered + observed_covered, 1e-9)
    )


def candidate_from_homography(template, page_gray, homography, inliers, good):
    height, width = template["crop"].shape
    uniform_scale = float(np.hypot(homography[0, 0], homography[0, 1]))
    if not 0.05 <= uniform_scale <= 2:
        return None
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return None
    x1, y1 = warped.min(axis=0)
    x2, y2 = warped.max(axis=0)
    quad_area = abs(cv2.contourArea(warped))
    if not (
        100 <= quad_area <= template["crop"].size * 4
        and 5 <= x2 - x1 <= page_gray.shape[1] / 2
        and 5 <= y2 - y1 <= page_gray.shape[0] / 2
        and x1 >= 0
        and y1 >= 0
        and x2 <= page_gray.shape[1]
        and y2 <= page_gray.shape[0]
    ):
        return None

    score = edge_similarity(template["crop"], page_gray, homography)
    if inliers >= 12 and score >= 0.40:
        status = "strong_candidate"
    elif inliers >= 8 and score >= 0.15:
        status = "review"
    else:
        return None

    factor = DPI / 72
    return {
        "class_name": template["class_name"],
        "source_image": template["image"],
        "status": status,
        "inliers": int(inliers),
        "feature_matches": int(good),
        "edge_score": round(score, 4),
        "bbox_pdf_pt": [
            round(float(x1 / factor), 2),
            round(float(y1 / factor), 2),
            round(float(x2 / factor), 2),
            round(float(y2 / factor), 2),
        ],
    }


def same_position(a, b):
    ax1, ay1, ax2, ay2 = a["bbox_pdf_pt"]
    bx1, by1, bx2, by2 = b["bbox_pdf_pt"]
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-9) >= 0.35


def main():
    for path in (PDF_PATH, IMAGE_ROOT, LABEL_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    sift = cv2.SIFT_create()
    templates = load_templates(sift)
    class_names = sorted({item["class_name"] for item in templates})
    print(f"기준 이미지 {len(templates)}장 / 클래스 {len(class_names)}개")

    with fitz.open(PDF_PATH) as doc:
        if not 0 <= PAGE_INDEX < len(doc):
            raise ValueError("PDF 페이지 번호를 확인하세요.")
        page = doc[PAGE_INDEX]
        if page.rotation != 0:
            raise ValueError("회전되지 않은 PDF 페이지를 기준으로 합니다.")
        factor = DPI / 72
        pix = page.get_pixmap(
            matrix=fitz.Matrix(factor, factor),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        page_gray = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width
        )
        page_keypoints, page_descriptors = sift.detectAndCompute(page_gray, None)
        if page_descriptors is None:
            raise ValueError("PDF에서 비교할 선 특징을 찾지 못했습니다.")
        print(f"PDF 특징점 {len(page_keypoints)}개")

        matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 64}
        )
        matcher.add([page_descriptors])
        matcher.train()
        candidates = []

        for number, template in enumerate(templates, start=1):
            descriptors = template["descriptors"]
            if descriptors is None or len(descriptors) < 4:
                continue
            pairs = matcher.knnMatch(descriptors, k=2)
            remaining = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < RATIO_TEST * pair[1].distance
            ]
            # shortcut: 같은 템플릿의 여러 실례를 RANSAC으로 반복 추출하지만
            # 반복·대칭 심볼은 특징점 비율 검사에서 빠질 수 있어 미검출 검수가 필요함.
            for _ in range(MAX_INSTANCES_PER_TEMPLATE):
                if len(remaining) < 8:
                    break
                source = np.float32(
                    [template["keypoints"][m.queryIdx].pt for m in remaining]
                ).reshape(-1, 1, 2)
                target = np.float32(
                    [page_keypoints[m.trainIdx].pt for m in remaining]
                ).reshape(-1, 1, 2)
                affine, mask = cv2.estimateAffinePartial2D(
                    source, target,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=RANSAC_ERROR,
                    maxIters=5000,
                )
                if affine is None or mask is None:
                    break
                homography = np.vstack([affine, [0, 0, 1]])
                inliers = int(mask.sum())
                if inliers >= 8:
                    found = candidate_from_homography(
                        template, page_gray, homography, inliers, len(remaining)
                    )
                    if found is not None:
                        candidates.append(found)
                remaining = [
                    match
                    for match, used in zip(remaining, mask.ravel())
                    if not used
                ]
            if number % 10 == 0 or number == len(templates):
                print(f"기준 이미지 {number}/{len(templates)} 검사 완료")

        candidates.sort(
            key=lambda c: (
                c["status"] == "strong_candidate",
                c["edge_score"],
                c["inliers"],
            ),
            reverse=True,
        )
        detections = []
        for candidate in candidates:
            if any(
                other["class_name"] == candidate["class_name"]
                and same_position(candidate, other)
                for other in detections
            ):
                continue
            detections.append(candidate)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        crop_dir = OUTPUT_DIR / "crops"
        crop_dir.mkdir()
        drawing = page.get_displaylist(annots=False)
        for number, result in enumerate(detections, start=1):
            result["id"] = number
            box = fitz.Rect(result["bbox_pdf_pt"])
            clip = fitz.Rect(
                box.x0 - 10, box.y0 - 10, box.x1 + 10, box.y1 + 10
            ) & page.rect
            crop_file = crop_dir / f"{number:03d}_{result['class_name']}.png"
            drawing.get_pixmap(
                matrix=fitz.Matrix(2, 2), clip=clip, alpha=False
            ).save(crop_file)
            result["crop_file"] = str(crop_file)
            color = (
                (0, 0.6, 0)
                if result["status"] == "strong_candidate"
                else (1, 0.55, 0)
            )
            page.draw_rect(box, color=color, width=1)
            page.insert_text(
                (box.x0, max(6, box.y0 - 3)),
                f"{result['class_name']} {result['edge_score']:.2f}",
                fontsize=6,
                color=color,
            )
        result_pdf = OUTPUT_DIR / "all_class_candidates.pdf"
        doc.save(result_pdf, garbage=4, deflate=True)

    by_class = defaultdict(list)
    for result in detections:
        by_class[result["class_name"]].append(result)
    summary = []
    for class_name in class_names:
        values = by_class[class_name]
        status = (
            "strong_candidate"
            if any(v["status"] == "strong_candidate" for v in values)
            else "review" if values else "not_detected"
        )
        summary.append(
            {
                "class_name": class_name,
                "status": status,
                "candidate_count": len(values),
            }
        )
    report = {
        "source_pdf": str(PDF_PATH),
        "page_number": PAGE_INDEX + 1,
        "model": "SIFT + geometric and edge matching; no YOLO",
        "template_count": len(templates),
        "class_count": len(class_names),
        "class_summary": summary,
        "detections": detections,
    }
    (OUTPUT_DIR / "all_class_candidates.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "all_class_candidates.csv").write_text(
        "class_name,status,candidate_count\n"
        + "".join(
            f"{s['class_name']},{s['status']},{s['candidate_count']}\n"
            for s in summary
        ),
        encoding="utf-8-sig",
    )
    columns, cell_width, cell_height = 4, 300, 250
    gallery = Image.new(
        "RGB",
        (columns * cell_width, max(1, math.ceil(len(detections) / columns)) * cell_height),
        "white",
    )
    painter = ImageDraw.Draw(gallery)
    for index, result in enumerate(detections):
        crop = Image.open(result["crop_file"]).convert("RGB")
        crop.thumbnail((cell_width - 18, cell_height - 45))
        left = index % columns * cell_width
        top = index // columns * cell_height
        gallery.paste(crop, (left + (cell_width - crop.width) // 2, top + 35))
        color = (0, 110, 0) if result["status"] == "strong_candidate" else (180, 100, 0)
        painter.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline=color,
            width=2,
        )
        painter.text(
            (left + 6, top + 5),
            f"{result['class_name']} {result['edge_score']:.2f} {result['status']}",
            fill=color,
        )
    gallery.save(OUTPUT_DIR / "candidate_gallery.png")
    print("클래스 상태:", dict(Counter(s["status"] for s in summary)))
    print("결과 폴더:", OUTPUT_DIR)
    for item in summary:
        print(item["class_name"], item["status"], item["candidate_count"])
    return OUTPUT_DIR


if __name__ == "__main__":
    main()
