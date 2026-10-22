"""Add the verified symbol-1 fitting to the checked 59-block sample DWG.

This is calibrated to the supplied PDF/DWG pair. The YOLO box includes a pipe;
only the inspected fitting rectangle is eligible for the new block.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pymupdf

import blockify_pdf_symbols as cad


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "source_inputs" / "DWG 객체 변환 샘플.pdf"
PRIOR = ROOT / "combined_shape_output" / "20261012_182144_178758" / "combined_candidates.json"
PREDICTIONS = ROOT / "symbol_1_finetune" / "pdf_detection_640" / "detections.json"
INPUT_DWG = ROOT / "verified_dwg" / "20261014" / "symbols_59_bom_title_fit.dwg"
OUTPUT = ROOT / "verified_dwg" / "20261016"
MERGED = ROOT / "combined_shape_output" / "20261016_class1_added"
FITTING_DWG = (31.45, 13.07, 31.70, 13.21)  # excludes vertical pipe and callout
BLOCK_NAME = "SYM_1_060"


def fitting_pdf_box(page_height: float) -> list[float]:
    x0, y0, x1, y1 = FITTING_DWG
    return [x0 * 72, page_height - y1 * 72, x1 * 72, page_height - y0 * 72]


def merge_detection(page_height: float) -> dict:
    previous = json.loads(PRIOR.read_text(encoding="utf-8"))
    predicted = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    hits = [row for row in predicted if row["class_name"] == "1"]
    if len(hits) != 1:
        raise ValueError(f"Expected one class-1 candidate, found {len(hits)}")
    model_box = hits[0]["pdf_bbox_xyxy_points"]
    fitting_box = fitting_pdf_box(page_height)
    if not (model_box[0] <= fitting_box[0] < fitting_box[2] <= model_box[2]
            and model_box[1] <= fitting_box[1] < fitting_box[3] <= model_box[3]):
        raise ValueError("The model detection does not contain the inspected fitting")
    if any(row["class_name"] == "1" for row in previous["displayed_detections"]):
        raise ValueError("Prior result already contains class 1")
    added = {"class_name": "1", "status": "strong_candidate",
             "search": "yolo26s-class1-verified", "confidence": hits[0]["confidence"],
             "bbox_pdf_pt": fitting_box, "model_bbox_pdf_pt": model_box,
             "source_window_px": hits[0]["source_window_px"]}
    previous["displayed_detections"].append(added)
    previous["source_pdf"] = str(PDF.resolve())
    previous["class_1_model"] = str((ROOT / "symbol_1_finetune" / "class_1" / "weights" / "best.pt").resolve())
    for row in previous["summary"]:
        if row["class_name"] == "1":
            row.update(candidate_count=1, status="strong_candidate")
            break
    else:
        previous["summary"].append({"class_name": "1", "candidate_count": 1,
                                    "status": "strong_candidate"})
    MERGED.mkdir(parents=True, exist_ok=True)
    (MERGED / "combined_candidates.json").write_text(
        json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    with (MERGED / "combined_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, ["class_name", "candidate_count", "status"])
        writer.writeheader()
        writer.writerows(previous["summary"])
    with pymupdf.open(PDF) as doc:
        page = doc[0]
        for row in previous["displayed_detections"]:
            box = pymupdf.Rect(row["bbox_pdf_pt"])
            color = (1, 0, 0) if row["class_name"] == "1" else (0, .45, 1)
            page.draw_rect(box, color=color, width=1.4)
            page.insert_text((box.x0, max(6, box.y0 - 4)), row["class_name"],
                             fontsize=6, color=color)
        doc.save(MERGED / "combined_candidates.pdf", garbage=4, deflate=True)
    return added


def main() -> Path:
    for path in (PDF, PRIOR, PREDICTIONS, INPUT_DWG, cad.CORE_CONSOLE):
        if not path.is_file():
            raise FileNotFoundError(path)
    with pymupdf.open(PDF) as doc:
        page_height = doc[0].rect.height
    new_detection = merge_detection(page_height)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "symbols_60_bom_title_fit.dwg"
    if destination.exists():
        if destination.read_bytes() != INPUT_DWG.read_bytes():
            raise FileExistsError(destination)
    else:
        shutil.copy2(INPUT_DWG, destination)
    before = cad.read_entities(cad.audit(destination, OUTPUT, "before"))
    x0, y0, x1, y1 = FITTING_DWG
    hits = [row for row in before
            if row["layer"] in cad.ALLOWED_LAYERS
            and row["type"] in cad.ALLOWED_TYPES
            and row["color"] in cad.ALLOWED_COLORS
            and x0 <= row["bounds"][0] and y0 <= row["bounds"][1]
            and row["bounds"][2] <= x1 and row["bounds"][3] <= y1]
    if len(hits) != 44:
        raise ValueError(f"Expected 44 fitting entities; found {len(hits)}. DWG unchanged.")
    existing = [row for row in before if row["type"] == "INSERT"
                and row["block_name"].startswith("SYM_")]
    if len(existing) != 59:
        raise ValueError(f"Expected 59 existing symbol blocks; found {len(existing)}")
    group = {"block_name": BLOCK_NAME, "definition_name": BLOCK_NAME,
             "class_name": "1", "status": "strong_candidate",
             "bbox_pdf_pt": new_detection["bbox_pdf_pt"], "bbox_dwg": FITTING_DWG,
             "handles": [row["handle"] for row in hits], "left_as_pipe": [],
             "entity_count": len(hits), "reuse_score": None, "reuse_scale": 1.0}
    cad.write_manifest([group], OUTPUT)
    script = cad.build_block_script([group], OUTPUT)
    cad.run_console(destination, script, OUTPUT / "create_blocks.log")
    after = cad.read_entities(cad.audit(destination, OUTPUT, "after"))
    verification = cad.verify(OUTPUT, before, after, [group], 0)
    remaining = [row for row in after if row["type"] == "INSERT"
                 and row["block_name"].startswith("SYM_")]
    if len(remaining) != 60 or sum(row["block_name"] == BLOCK_NAME for row in remaining) != 1:
        raise RuntimeError("Expected 60 independent symbol INSERTs after saving")
    verification.update(existing_symbol_blocks=59, final_symbol_blocks=60,
                        fitting_entities=44, input_dwg=str(INPUT_DWG),
                        output_dwg=str(destination),
                        excluded_pipe_geometry=True)
    (OUTPUT / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)
    return destination


if __name__ == "__main__":
    main()
