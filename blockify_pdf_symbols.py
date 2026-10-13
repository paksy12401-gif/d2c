"""Turn detected PDF symbol locations into reusable blocks in an imported DWG.

This script is specific to the supplied PDF / Drawing1.dwg pair. It copies the
DWG, audits its vector entities with AutoCAD Core Console, and replaces only
symbol geometry in detected boxes with BLOCK references. The audited pipe
outlines and through-running harness lines remain independent CAD entities. Identical
looking, same-class instances may share one block definition.
The source DWG is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import pymupdf


ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PDF_PATH = Path(r"C:\Users\Win11Pro\Desktop\1. DWG 객체 변환 도면 생성 샘플\DWG 객체 변환 샘플.pdf")
DWG_PATH = Path(r"C:\Users\Win11Pro\Desktop\Drawing1.dwg")
DETECTIONS_JSON = ROOT / "combined_shape_output" / "20261012_182144_178758" / "combined_candidates.json"
CORE_CONSOLE = Path(r"C:\Program Files\Autodesk\AutoCAD 2027\accoreconsole.exe")
OUTPUT_ROOT = ROOT / "blockified_dwg"
PIPELINE_VERSION = "symbol-blocks-v9-exact-geometry"
INCLUDE_REVIEW = True  # Include the 8 review candidates as requested.
LOCK_PDF_BACKGROUND = False  # Keep remaining geometry editable, as in the checked DWG.
PDF_TO_DWG_SCALE = 1.0 / 72.0  # Confirmed for this PDFIMPORT drawing.
BBOX_PADDING_PT = 0.0
# CAD-unit margins for fittings whose detected boxes crop their mounting parts.
# Order: left, bottom, right, top. These values are calibrated to this drawing.
SYMBOL_BOX_EXPANSION_DWG = {
    "14": (0.20, 0.15, 0.20, 0.15),
    "7": (0.22, 0.10, 0.22, 0.10),
    "13": (0.30, 0.32, 0.30, 0.32),
    "11": (0.04, 0.16, 0.04, 0.02),
}
# Source-DWG handles of short pipe/sleeve outlines overlapping symbol boxes.
# These were checked against this exact Drawing1.dwg; the source DWG is fixed.
# The fasteners and connector fittings next to them remain in their blocks.
PIPE_COMPONENT_HANDLES_BY_BLOCK = {
    "SYM_7_014": ("7DB", "828", "18F9"),
    "SYM_13_022": ("749", "7AE"),
    "SYM_7_028": ("7F4",),
    "SYM_7_029": ("767",),
    "SYM_14_031": ("6B5", "1812", "1870", "1D20", "78B"),
    "SYM_13_034": ("5FC", "18A3"),
    "SYM_7_036": ("67B",),
    "SYM_11_037": ("7B8", "775", "185B", "185C"),
    "SYM_7_039": ("71F",),
    "SYM_7_040": ("633",),
    "SYM_14_041": ("74B", "184E", "18F8", "21BD", "74A"),
    "SYM_14_043": ("804",),
    "SYM_7_044": ("7F5",),
    "SYM_11_045": ("605", "6A5", "180C"),
    "SYM_7_046": ("697",),
    "SYM_11_047": ("6B2", "850", "1905"),
    "SYM_14_048": ("695", "185A", "773"),
    "SYM_13_049": ("7D3",),
    "SYM_14_050": ("6D3", "181A", "6D2"),
    "SYM_14_051": ("7D6", "6E7", "181E", "181F", "1877", "6E6"),
}
ALLOWED_LAYERS = {"PDF_Geomertry", "PDF_Geometry"}
ALLOWED_TYPES = {"LWPOLYLINE", "CIRCLE", "ARC", "ELLIPSE"}
ALLOWED_COLORS = {"7", "60", "256"}  # Exclude green/red wire centerlines and callout colors.


AUDIT_SCRIPT = r'''
(setq fp (open (strcat (getvar "DWGPREFIX") "entity_audit.csv") "w"))
(write-line "handle,type,layer,color,minx,miny,maxx,maxy,cx,cy,r,ax,ay,ratio,start,end,block_name" fp)
(setq ss (ssget "_X" '((410 . "Model"))))
(setq i 0)
(while (and ss (< i (sslength ss)))
  (setq e (ssname ss i))
  (if e
    (progn
      (setq ed (entget e))
      (setq ty (cdr (assoc 0 ed)))
      (setq xmin nil ymin nil xmax nil ymax nil)
      (foreach pair ed
        (if (= (car pair) 10)
          (progn
            (setq pt (cdr pair))
            (if (and (listp pt) (numberp (car pt)) (numberp (cadr pt)))
              (progn
                (setq x (car pt) y (cadr pt))
                (setq xmin (if xmin (min xmin x) x))
                (setq xmax (if xmax (max xmax x) x))
                (setq ymin (if ymin (min ymin y) y))
                (setq ymax (if ymax (max ymax y) y)))))))
      (if xmin
        (write-line
          (strcat
            (cdr (assoc 5 ed)) "," ty "," (cdr (assoc 8 ed)) ","
            (itoa (if (assoc 62 ed) (cdr (assoc 62 ed)) 256)) ","
            (rtos xmin 2 9) "," (rtos ymin 2 9) "," (rtos xmax 2 9) "," (rtos ymax 2 9) ","
            (if (assoc 10 ed) (rtos (car (cdr (assoc 10 ed))) 2 9) "") ","
            (if (assoc 10 ed) (rtos (cadr (cdr (assoc 10 ed))) 2 9) "") ","
            (if (assoc 40 ed) (rtos (cdr (assoc 40 ed)) 2 9) "") ","
            (if (assoc 11 ed) (rtos (car (cdr (assoc 11 ed))) 2 9) "") ","
            (if (assoc 11 ed) (rtos (cadr (cdr (assoc 11 ed))) 2 9) "") ","
            (if (and (= ty "ELLIPSE") (assoc 40 ed)) (rtos (cdr (assoc 40 ed)) 2 9) "") ","
            (if (assoc 50 ed) (rtos (cdr (assoc 50 ed)) 2 9)
                (if (and (= ty "ELLIPSE") (assoc 41 ed)) (rtos (cdr (assoc 41 ed)) 2 9) "")) ","
            (if (assoc 51 ed) (rtos (cdr (assoc 51 ed)) 2 9)
                (if (and (= ty "ELLIPSE") (assoc 42 ed)) (rtos (cdr (assoc 42 ed)) 2 9) "")) ","
            (if (and (= ty "INSERT") (assoc 2 ed)) (cdr (assoc 2 ed)) ""))
          fp))))
  (setq i (1+ i)))
(close fp)
(princ (strcat "\nAUDIT_ROWS=" (itoa i)))
(princ)
_.QUIT
_N
'''.strip() + "\n"


def _number(value: str) -> float:
    return float(value) if value else float("nan")


def _ccw_angles(start: float, end: float, critical: list[float]) -> list[float]:
    two_pi = 2 * math.pi
    while end < start:
        end += two_pi
    values = [start, end]
    for angle in critical:
        for turn in range(-2, 4):
            test = angle + turn * two_pi
            if start <= test <= end:
                values.append(test)
    return values


def entity_bounds(row: dict[str, str]) -> tuple[float, float, float, float]:
    kind = row["type"]
    if kind == "LWPOLYLINE" or kind == "INSERT":
        return tuple(float(row[key]) for key in ("minx", "miny", "maxx", "maxy"))
    cx, cy = float(row["cx"]), float(row["cy"])
    if kind == "CIRCLE":
        radius = float(row["r"])
        return cx - radius, cy - radius, cx + radius, cy + radius
    if kind == "ARC":
        radius = float(row["r"])
        # AUDIT_SCRIPT exports entget angles, which are already in radians.
        start, end = float(row["start"]), float(row["end"])
        angles = _ccw_angles(start, end, [n * math.pi / 2 for n in range(4)])
        xs = [cx + radius * math.cos(a) for a in angles]
        ys = [cy + radius * math.sin(a) for a in angles]
        return min(xs), min(ys), max(xs), max(ys)
    if kind == "ELLIPSE":
        ax, ay, ratio = float(row["ax"]), float(row["ay"]), float(row["ratio"])
        start, end = float(row["start"]), float(row["end"])
        x_phase = math.atan2(-ay * ratio, ax)
        y_phase = math.atan2(ax * ratio, ay)
        angles = _ccw_angles(start, end, [x_phase, x_phase + math.pi, y_phase, y_phase + math.pi])
        xs = [cx + ax * math.cos(a) - ay * ratio * math.sin(a) for a in angles]
        ys = [cy + ay * math.cos(a) + ax * ratio * math.sin(a) for a in angles]
        return min(xs), min(ys), max(xs), max(ys)
    raise ValueError(f"Unsupported entity type: {kind}")


def run_console(dwg_path: Path, script_path: Path, log_path: Path) -> None:
    with log_path.open("wb") as stream:
        process = subprocess.run(
            [str(CORE_CONSOLE), "/i", str(dwg_path), "/s", str(script_path), "/l", "en-US"],
            stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=False,
        )
    if process.returncode:
        raise RuntimeError(f"AutoCAD Core Console exit {process.returncode}: {log_path}")


def audit(dwg_path: Path, folder: Path, suffix: str) -> Path:
    script = folder / f"audit_{suffix}.scr"
    script.write_text(AUDIT_SCRIPT, encoding="ascii")
    output = folder / "entity_audit.csv"
    if output.exists():
        output.unlink()
    run_console(dwg_path, script, folder / f"audit_{suffix}.log")
    if not output.is_file() or output.stat().st_size < 100:
        raise RuntimeError("DWG 엔티티 감사 CSV가 생성되지 않았습니다.")
    final = folder / f"entities_{suffix}.csv"
    output.rename(final)
    return final


def read_entities(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["type"] in ALLOWED_TYPES:
            row["bounds"] = entity_bounds(row)
    return rows


def pdf_box_to_dwg(bbox: list[float], page_height_pt: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    pad = BBOX_PADDING_PT
    scale = PDF_TO_DWG_SCALE
    return (
        (x1 - pad) * scale,
        (page_height_pt - y2 - pad) * scale,
        (x2 + pad) * scale,
        (page_height_pt - y1 + pad) * scale,
    )


def assign_entities(detections: list[dict], entities: list[dict], page_height_pt: float):
    groups = []
    candidates = [
        row for row in detections
        if row["status"] == "strong_candidate" or INCLUDE_REVIEW
    ]
    for index, detection in enumerate(candidates, 1):
        detector_box = pdf_box_to_dwg(detection["bbox_pdf_pt"], page_height_pt)
        left, bottom, right, top = SYMBOL_BOX_EXPANSION_DWG.get(
            detection["class_name"], (0, 0, 0, 0)
        )
        if detection["class_name"] == "14" and index == 51:
            top = 0.75  # This rotated clamp's upper mounting piece is farther out.
        box = (detector_box[0] - left, detector_box[1] - bottom,
               detector_box[2] + right, detector_box[3] + top)
        x1, y1, x2, y2 = box
        hits = [row for row in entities
                if row["layer"] in ALLOWED_LAYERS and row["type"] in ALLOWED_TYPES
                and row["color"] in ALLOWED_COLORS
                and row["bounds"][0] >= x1 and row["bounds"][1] >= y1
                and row["bounds"][2] <= x2 and row["bounds"][3] <= y2]
        name_part = re.sub(r"[^A-Za-z0-9_]", "_", detection["class_name"])
        block_name = f"SYM_{name_part}_{index:03d}"
        pipe_handles = set(PIPE_COMPONENT_HANDLES_BY_BLOCK.get(block_name, ()))
        pipe_found = {row["handle"] for row in hits if row["handle"] in pipe_handles}
        if pipe_found != pipe_handles:
            raise ValueError(f"Audited pipe handles missing from {block_name}: {pipe_handles - pipe_found}")
        # A former "edge-to-edge wire" heuristic mistook broad fastener flanges
        # for pipes. Exclude only audited pipe outlines; long pipe rails are
        # already outside these compact symbol boxes.
        hits = [row for row in hits if row["handle"] not in pipe_handles]
        groups.append({
            "block_name": block_name,
            "class_name": detection["class_name"],
            "status": detection["status"],
            "bbox_pdf_pt": detection["bbox_pdf_pt"],
            "bbox_dwg": box,
            "handles": [row["handle"] for row in hits],
            "left_as_pipe": sorted(pipe_handles),
            "entity_count": len(hits),
        })
    membership = defaultdict(list)
    for group in groups:
        for handle in group["handles"]:
            membership[handle].append(group["block_name"])
    conflicts = {handle: names for handle, names in membership.items() if len(names) > 1}
    if conflicts:
        raise ValueError(f"한 엔티티가 여러 심볼 박스에 포함됨: {len(conflicts)}개. 자동 변환 중단.")
    return groups


def plan_definition_reuse(groups: list[dict], pdf_path: Path) -> None:
    """Use each location's exact vectors; similar raster crops can differ by a line."""
    for group in groups:
        group["definition_name"] = group["block_name"]
        group["reuse_score"] = None
        group["reuse_scale"] = 1.0


def build_block_script(groups: list[dict], folder: Path) -> Path:
    lines = [
        '(setvar "CMDECHO" 0)',
        '(setvar "FILEDIA" 0)',
        '(setvar "BLOCKCREATEMODE" 0)',
        '(setq fp (open (strcat (getvar "DWGPREFIX") "block_command_report.csv") "w"))',
        '(write-line "block_name,selected,created,definition_name,reused" fp)',
    ]
    for group in groups:
        if group["entity_count"] == 0:
            continue
        lines.append("(setq ss (ssadd))")
        lines.extend(
            f'(if (handent "{handle}") (ssadd (handent "{handle}") ss))'
            for handle in group["handles"]
        )
        x1, y1, x2, y2 = group["bbox_dwg"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        name = group["block_name"]
        definition = group["definition_name"]
        if definition == name:
            lines.append(
                f'(command "_.-BLOCK" "{name}" "_Mode" "_Convert" '
                f'(list {cx:.9f} {cy:.9f} 0.0) ss "")'
            )
            lines.append(
                f'(write-line (strcat "{name}," (itoa (sslength ss)) "," '
                f'(if (tblsearch "BLOCK" "{name}") "1" "0") "," "{definition},0") fp)'
            )
        else:
            scale = group["reuse_scale"]
            lines.append(
                f'(setq created (entmakex (list (cons 0 "INSERT") (cons 2 "{definition}") '
                f'(cons 8 "0") (cons 10 (list {cx:.9f} {cy:.9f} 0.0)) '
                f'(cons 41 {scale:.9f}) (cons 42 {scale:.9f}) (cons 43 1.0) (cons 50 0.0))))'
            )
            lines.append('(if created (command "_.ERASE" ss ""))')
            lines.append(
                f'(write-line (strcat "{name}," (itoa (sslength ss)) "," '
                f'(if created "1" "0") "," "{definition},1") fp)'
            )
    lines.append('(close fp)')
    if LOCK_PDF_BACKGROUND:
        lines.extend([
            '(foreach layer \'("PDF_Geomertry" "PDF_Geometry" "PDF_Text" "PDF_Solid Fills")',
            '  (setq le (tblobjname "LAYER" layer))',
            '  (if le (progn (setq ld (entget le))',
            '    (entmod (subst (cons 70 (logior (cdr (assoc 70 ld)) 4)) (assoc 70 ld) ld)))))',
        ])
    lines.extend(['(command "_.QSAVE")', '(princ)', '_.QUIT', '_N'])
    path = folder / "create_blocks.scr"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_manifest(groups: list[dict], folder: Path) -> None:
    fields = ["block_name", "class_name", "status", "entity_count", "left_as_pipe",
              "definition_name", "reuse_score", "reuse_scale", "bbox_pdf_pt", "bbox_dwg"]
    with (folder / "block_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: group[key] for key in fields} for group in groups)
    (folder / "block_manifest.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def verify(folder: Path, before: list[dict], after: list[dict], groups: list[dict], skipped_review_boxes: int) -> dict:
    report_path = folder / "block_command_report.csv"
    if not report_path.is_file():
        raise RuntimeError("AutoCAD 블록 생성 결과표가 없습니다.")
    with report_path.open(encoding="utf-8", newline="") as handle:
        command_results = list(csv.DictReader(handle))
    requested = [group for group in groups if group["entity_count"] > 0]
    if len(command_results) != len(requested) or any(row["created"] != "1" for row in command_results):
        raise RuntimeError("일부 BLOCK 명령이 실패했습니다. block_command_report.csv를 확인하세요.")
    before_handles = {row["handle"] for row in before}
    after_handles = {row["handle"] for row in after}
    removed = before_handles - after_handles
    added = after_handles - before_handles
    expected_removed = {handle for group in requested for handle in group["handles"]}
    new_inserts = [row for row in after if row["handle"] in added and row["type"] == "INSERT"]
    actual_definitions = Counter(row["block_name"] for row in new_inserts)
    expected_definitions = Counter(group["definition_name"] for group in requested)
    if removed != expected_removed or len(new_inserts) != len(requested):
        raise RuntimeError(
            f"변환 후 엔티티 검증 실패: 제거 {len(removed)}/{len(expected_removed)}, "
            f"새 INSERT {len(new_inserts)}/{len(requested)}. 결과 폴더를 확인하세요."
        )
    if actual_definitions != expected_definitions:
        raise RuntimeError(f"블록 정의 참조가 계획과 다릅니다: {actual_definitions} / {expected_definitions}")
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "input_model_entities": len(before),
        "converted_source_entities": len(expected_removed),
        "new_block_references": len(new_inserts),
        "unique_block_definitions": len(actual_definitions),
        "reused_block_references": sum(group["definition_name"] != group["block_name"] for group in requested),
        "pipe_entities_left_unblocked": sum(len(group["left_as_pipe"]) for group in groups),
        "pdf_background_layers_locked": LOCK_PDF_BACKGROUND,
        "output_model_entities": len(after),
        "skipped_review_boxes": skipped_review_boxes,
        "included_review_boxes": sum(group["status"] != "strong_candidate" for group in requested),
    }
    (folder / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> Path:
    for path in (PDF_PATH, DWG_PATH, DETECTIONS_JSON, CORE_CONSOLE):
        if not path.is_file():
            raise FileNotFoundError(path)
    with pymupdf.open(PDF_PATH) as doc:
        if len(doc) != 1:
            raise ValueError("현재 코드는 1페이지 PDF만 처리합니다.")
        page_width, page_height = doc[0].rect.width, doc[0].rect.height
    detection_report = json.loads(DETECTIONS_JSON.read_text(encoding="utf-8"))
    if Path(detection_report["source_pdf"]).resolve() != PDF_PATH.resolve():
        raise ValueError("검출 JSON과 PDF 원본이 서로 다릅니다.")
    run_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    output_dwg = run_dir / "Drawing1_symbols_blockified.dwg"
    shutil.copy2(DWG_PATH, output_dwg)
    before_path = audit(output_dwg, run_dir, "before")
    before = read_entities(before_path)
    max_x = max(float(row["maxx"]) for row in before)
    max_y = max(float(row["maxy"]) for row in before)
    if abs(max_x - page_width * PDF_TO_DWG_SCALE) > 0.2 or abs(max_y - page_height * PDF_TO_DWG_SCALE) > 0.2:
        raise ValueError("PDF와 DWG의 좌표 범위가 맞지 않습니다. 변환을 중단했습니다.")
    groups = assign_entities(detection_report["displayed_detections"], before, page_height)
    plan_definition_reuse(groups, PDF_PATH)
    write_manifest(groups, run_dir)
    print(f"원본 보존: {DWG_PATH}")
    print(f"심볼 {len(groups)}곳, 대상 엔티티 {sum(group['entity_count'] for group in groups)}개")
    script = build_block_script(groups, run_dir)
    run_console(output_dwg, script, run_dir / "create_blocks.log")
    after_path = audit(output_dwg, run_dir, "after")
    after = read_entities(after_path)
    skipped_reviews = sum(
        row["status"] != "strong_candidate"
        for row in detection_report["displayed_detections"]
    ) if not INCLUDE_REVIEW else 0
    verification = verify(run_dir, before, after, groups, skipped_reviews)
    print("검증:", verification)
    print("결과 DWG:", output_dwg)
    return output_dwg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-review", action=argparse.BooleanOptionalAction,
                        default=INCLUDE_REVIEW, help="검토 필요 후보 8곳도 블록으로 변환 (기본: 포함)")
    args = parser.parse_args()
    INCLUDE_REVIEW = args.include_review
    main()
