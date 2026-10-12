"""Combine coarse and high-resolution symbol searches on one clean PDF page.

Overlapping candidates are displayed once. The omitted candidate remains in the
JSON/CSV conflict report so a person can review possible class confusion.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json

import pymupdf


WORKSPACE = Path(__file__).resolve().parent
COARSE_JSON = WORKSPACE / "shape_match_all_output" / "20261012_174753_645724" / "all_class_candidates.json"
FINE_JSON = WORKSPACE / "small_symbol_highres_output" / "20261012_181616_777313" / "small_symbol_candidates.json"
OUTPUT_ROOT = WORKSPACE / "combined_shape_output"


def overlap(a, b):
    ax1, ay1, ax2, ay2 = a["bbox_pdf_pt"]
    bx1, by1, bx2, by2 = b["bbox_pdf_pt"]
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    union = area_a + area_b - intersection
    return intersection / max(union, 1e-9), intersection / max(min(area_a, area_b), 1e-9)


def merge_candidates(coarse, fine):
    all_rows = [dict(row, search="72dpi") for row in coarse]
    all_rows += [dict(row, search="288dpi") for row in fine]
    # Strong candidates take display precedence over review candidates.
    # A tie keeps the previously accepted 72 DPI result; the alternative is recorded.
    all_rows.sort(key=lambda row: (row["status"] != "strong_candidate", row["search"] != "72dpi"))
    displayed = []
    conflicts = []
    for row in all_rows:
        duplicate = None
        for kept in displayed:
            iou, containment = overlap(row, kept)
            if iou >= 0.35 or containment >= 0.80:
                duplicate = kept, iou, containment
                break
        if duplicate is None:
            displayed.append(row)
        else:
            kept, iou, containment = duplicate
            conflicts.append({
                "displayed_class": kept["class_name"],
                "displayed_search": kept["search"],
                "omitted_class": row["class_name"],
                "omitted_search": row["search"],
                "omitted_status": row["status"],
                "iou": round(iou, 3),
                "containment": round(containment, 3),
                "displayed_bbox_pdf_pt": kept["bbox_pdf_pt"],
                "omitted_bbox_pdf_pt": row["bbox_pdf_pt"],
            })
    return displayed, conflicts


def main(coarse_path=COARSE_JSON, fine_path=FINE_JSON):
    coarse_report = json.loads(Path(coarse_path).read_text(encoding="utf-8"))
    fine_report = json.loads(Path(fine_path).read_text(encoding="utf-8"))
    pdf_path = Path(coarse_report["source_pdf"])
    if pdf_path != Path(fine_report["source_pdf"]):
        raise ValueError("두 결과가 서로 다른 PDF를 사용했습니다.")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    displayed, conflicts = merge_candidates(
        coarse_report["detections"], fine_report["detections"]
    )
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    with pymupdf.open(pdf_path) as doc:
        page = doc[0]
        for row in displayed:
            box = pymupdf.Rect(row["bbox_pdf_pt"])
            color = (0, 0.25, 1) if row["search"] == "72dpi" else (0, 0.6, 0)
            page.draw_rect(box, color=color, width=1.4)
            source = "L" if row["search"] == "72dpi" else "S"
            page.insert_text(
                (box.x0, max(6, box.y0 - 4)),
                f"{row['class_name']} [{source}]",
                fontsize=6, color=color,
            )
        doc.save(output_dir / "combined_candidates.pdf", garbage=4, deflate=True)
    all_names = sorted({item["class_name"] for item in coarse_report["class_summary"]}
                       | {item["class_name"] for item in fine_report["summary"]})
    by_class = defaultdict(list)
    for row in displayed:
        by_class[row["class_name"]].append(row)
    summary = []
    for name in all_names:
        rows = by_class[name]
        summary.append({
            "class_name": name,
            "candidate_count": len(rows),
            "status": "strong_candidate" if any(r["status"] == "strong_candidate" for r in rows)
            else "review" if rows else "not_detected",
        })
    with (output_dir / "combined_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "candidate_count", "status"])
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "overlap_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["displayed_class", "displayed_search", "omitted_class", "omitted_search",
                  "omitted_status", "iou", "containment", "displayed_bbox_pdf_pt", "omitted_bbox_pdf_pt"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(conflicts)
    (output_dir / "combined_candidates.json").write_text(
        json.dumps({"source_pdf": str(pdf_path), "coarse_json": str(coarse_path),
                    "fine_json": str(fine_path), "displayed_detections": displayed,
                    "overlap_review": conflicts, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"기존 {len(coarse_report['detections'])}개 + 고해상도 {len(fine_report['detections'])}개")
    print(f"PDF 표시 {len(displayed)}개 / 겹침 검토 {len(conflicts)}건")
    print("결과 폴더:", output_dir)
    return output_dir


if __name__ == "__main__":
    main()
