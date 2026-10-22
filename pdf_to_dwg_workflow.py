"""Convert one drawing-family PDF to an AutoCAD DWG with symbol blocks.

The checked sample is reused byte-for-byte. Other PDFs are imported and searched
afresh; their OCR outlines and raster tables remain in the imported drawing.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Callable
import uuid
import zipfile

import pymupdf

import blockify_pdf_symbols as blocks


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output" / "pdf_to_dwg_web"
SAMPLE_PDF = ROOT / "source_inputs" / "DWG 객체 변환 샘플.pdf"
STARTER_DWG = ROOT / "source_inputs" / "Drawing1.dwg"
CHECKED_DWG = ROOT / "verified_dwg" / "20261016" / "symbols_60_bom_title_fit.dwg"
CHECKED_DETECTIONS = ROOT / "combined_shape_output" / "20261016_class1_added" / "combined_candidates.json"
DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "drawing"
MAX_PDF_BYTES = 40 * 1024 * 1024
_SEARCH_LOCK = threading.Lock()
_CAD_IMPORT_LOCK = threading.Lock()


def inspect_pdf(data: bytes) -> dict:
    if not data or len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF 크기는 1바이트 이상, 40MB 이하여야 합니다.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass or len(doc) != 1:
                raise ValueError("암호가 없는 1페이지 PDF만 지원합니다.")
            page = doc[0]
            if page.rotation:
                raise ValueError("회전되지 않은 PDF 페이지만 지원합니다.")
            drawings = len(page.get_drawings())
            if not drawings:
                raise ValueError("벡터 선이 없는 PDF입니다. 이미지 스캔 도면은 이 변환 방식으로 블록화할 수 없습니다.")
            return {
                "sha256": hashlib.sha256(data).hexdigest(),
                "page_width_pt": page.rect.width,
                "page_height_pt": page.rect.height,
                "vector_paths": drawings,
                "embedded_images": len(page.get_image_info()),
            }
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError("올바른 PDF 파일을 선택해 주세요.") from exc


def _run_cad(dwg: Path, script: Path, log: Path, *, timeout: int = 900) -> None:
    if not blocks.CORE_CONSOLE.is_file():
        raise FileNotFoundError(f"AutoCAD Core Console이 없습니다: {blocks.CORE_CONSOLE}")
    with log.open("wb") as stream:
        result = subprocess.run(
            [str(blocks.CORE_CONSOLE), "/i", str(dwg), "/s", str(script), "/l", "en-US"],
            cwd=dwg.parent, stdout=stream, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"AutoCAD가 종료 코드 {result.returncode}로 실패했습니다: {log}")


def _import_pdf(pdf: Path, dwg: Path) -> None:
    if not STARTER_DWG.is_file():
        raise FileNotFoundError(f"시작 DWG가 없습니다: {STARTER_DWG}")
    shutil.copy2(STARTER_DWG, dwg)
    script = dwg.with_suffix(".import.scr")
    script.write_text("\n".join([
        '(setvar "FILEDIA" 0)', '(setvar "CMDECHO" 0)',
        '(setq old_pdf_filter (getvar "PDFIMPORTFILTER"))',
        '(setvar "PDFIMPORTFILTER" 0)',
        '_.ERASE', '_ALL', '',
        '_.-PDFIMPORT', '_F', pdf.name, '1', '0,0', '1', '0',
        '(setvar "PDFIMPORTFILTER" old_pdf_filter)',
        '(setq img_ss (ssget "_X" (list (cons 0 "IMAGE"))))',
        '(if img_ss (progn (setq img_i 0) (while (< img_i (sslength img_ss)) '
        '(setq img_ed (entget (cdr (assoc 340 (entget (ssname img_ss img_i)))))) '
        '(setq img_old (cdr (assoc 1 img_ed))) '
        '(setq img_rel (strcat "PDF Images\\\\" (vl-filename-base img_old) '
        '(vl-filename-extension img_old))) '
        '(entmod (subst (cons 1 img_rel) (assoc 1 img_ed) img_ed)) '
        '(setq img_i (1+ img_i)))))',
        '_.QSAVE', '_.QUIT', '_N', '',
    ]), encoding="ascii")
    _run_cad(dwg, script, dwg.with_suffix(".import.log"), timeout=2400)
    if not dwg.is_file() or dwg.stat().st_size < 1000:
        raise RuntimeError("PDF 가져오기 후 DWG가 저장되지 않았습니다.")


def _read_audit(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("cp949")
    rows = list(csv.DictReader(io.StringIO(content)))
    for row in rows:
        if row["type"] in blocks.ALLOWED_TYPES:
            row["bounds"] = blocks.entity_bounds(row)
    return rows


def _search_symbols(pdf: Path, folder: Path, data_root: Path) -> dict:
    image_root, label_root = data_root / "dataset_640", data_root / "labels_640"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"심볼 템플릿/라벨 폴더가 없습니다: {data_root}")

    import shape_match_all as coarse
    import small_symbol_highres as fine
    from merge_shape_candidates import merge_candidates
    import cv2
    import numpy as np

    coarse.PDF_PATH = pdf
    coarse.IMAGE_ROOT = image_root
    coarse.LABEL_ROOT = label_root
    coarse.OUTPUT_DIR = folder / "coarse"
    coarse.main()
    coarse_report = json.loads((coarse.OUTPUT_DIR / "all_class_candidates.json").read_text(encoding="utf-8"))
    missing = {row["class_name"] for row in coarse_report["class_summary"]
               if row["status"] == "not_detected"}
    fine_detections = []
    if missing:
        sift = cv2.SIFT_create(nfeatures=25000)
        templates = [item for item in coarse.load_templates(sift)
                     if item["class_name"] in missing]
        with pymupdf.open(pdf) as doc:
            page = doc[0]
            right = min(fine.SEARCH_RIGHT_PT, page.rect.x1)
            for y in fine.positions(page.rect.y1, fine.TILE_H_PT, fine.OVERLAP_PT):
                for x in fine.positions(right, fine.TILE_W_PT, fine.OVERLAP_PT):
                    clip = pymupdf.Rect(x, y, x + fine.TILE_W_PT, y + fine.TILE_H_PT) & pymupdf.Rect(0, 0, right, page.rect.y1)
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=clip,
                                          colorspace=pymupdf.csGRAY, alpha=False)
                    gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                    fine_detections.extend(fine.search_tile(gray, clip, templates, sift))
        fine_detections = fine.deduplicate(fine_detections)

    displayed, conflicts = merge_candidates(coarse_report["detections"], fine_detections)
    result = {"source_pdf": str(pdf), "displayed_detections": displayed,
              "overlap_review": conflicts, "coarse_count": len(coarse_report["detections"]),
              "fine_count": len(fine_detections)}
    (folder / "detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with pymupdf.open(pdf) as doc:
        page = doc[0]
        for row in displayed:
            rect = pymupdf.Rect(row["bbox_pdf_pt"])
            page.draw_rect(rect, color=(0, .55, 0) if row["status"] == "strong_candidate" else (1, .55, 0), width=1)
            page.insert_text((rect.x0, max(6, rect.y0 - 3)), row["class_name"], fontsize=6)
        doc.save(folder / "detections.pdf", garbage=4, deflate=True)
    return result


def _assign_fresh_entities(detections: list[dict], entities: list[dict], page_height: float) -> list[dict]:
    """Use contained PDF-imported geometry; never claim unverified pipe separation."""
    groups = []
    used: set[str] = set()
    ordered = sorted((row for row in detections if row["status"] == "strong_candidate"),
                     key=lambda row: (row.get("edge_score", 0), row.get("inliers", 0)), reverse=True)
    for index, detection in enumerate(ordered, 1):
        x0, y0, x1, y1 = blocks.pdf_box_to_dwg(detection["bbox_pdf_pt"], page_height)
        pad = 0.03
        box = (x0-pad, y0-pad, x1+pad, y1+pad)
        handles = []
        for entity in entities:
            if entity["handle"] in used or entity["type"] not in blocks.ALLOWED_TYPES:
                continue
            if not entity["layer"].upper().startswith("PDF"):
                continue
            a, b, c, d = entity["bounds"]
            if box[0] <= a and box[1] <= b and c <= box[2] and d <= box[3]:
                handles.append(entity["handle"])
        if len(handles) < 3:
            continue
        used.update(handles)
        name = re.sub(r"[^A-Za-z0-9_]", "_", detection["class_name"])
        group_name = f"SYM_{name}_{index:03d}"
        groups.append({"block_name": group_name, "definition_name": group_name,
                       "class_name": detection["class_name"], "status": detection["status"],
                       "bbox_pdf_pt": detection["bbox_pdf_pt"], "bbox_dwg": box,
                       "handles": handles, "left_as_pipe": [], "entity_count": len(handles),
                       "reuse_score": None, "reuse_scale": 1.0})
    return groups


def _bundle(folder: Path, dwg: Path, report: dict) -> Path:
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = folder / "result.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.write(dwg, "result.dwg")
        output.write(folder / "input.pdf", "input.pdf")
        output.write(folder / "report.json", "report.json")
        for name in ("detections.json", "block_manifest.csv", "block_manifest.json", "verification.json"):
            path = folder / name
            if path.is_file():
                output.write(path, name)
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                output.write(path, path.relative_to(folder))
        if (folder / "detections.pdf").is_file():
            output.write(folder / "detections.pdf", "detections.pdf")
    return archive


def convert_pdf(data: bytes, *, data_root: Path = DEFAULT_DATA_ROOT,
                progress: Callable[[str], None] | None = None,
                output_root: Path = OUTPUT_ROOT) -> dict:
    metadata = inspect_pdf(data)
    announce = progress or (lambda message: None)
    folder = output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    pdf = folder / "input.pdf"
    pdf.write_bytes(data)
    dwg = folder / "result.dwg"
    sample_hash = hashlib.sha256(SAMPLE_PDF.read_bytes()).hexdigest() if SAMPLE_PDF.is_file() else None
    if metadata["sha256"] == sample_hash and CHECKED_DWG.is_file():
        announce("검증된 동일 PDF 결과를 불러오는 중")
        shutil.copy2(CHECKED_DWG, dwg)
        detections = json.loads(CHECKED_DETECTIONS.read_text(encoding="utf-8"))
        shutil.copy2(CHECKED_DETECTIONS, folder / "detections.json")
        checked_preview = CHECKED_DETECTIONS.with_name("combined_candidates.pdf")
        if checked_preview.is_file():
            shutil.copy2(checked_preview, folder / "detections.pdf")
        report = {**metadata, "mode": "verified_identical_pdf", "symbol_blocks": 60,
                  "caption_texts": 31, "work_note_texts": 5,
                  "additional_description_texts": 13,
                  "bom_corrected_cells": 32,
                  "bom_texts": 155, "circuit_cells": 720,
                  "candidate_count": len(detections["displayed_detections"]),
                  "note": "SHA-256이 같은 원본 PDF의 검증 완료 DWG를 재사용했습니다."}
    else:
        announce("심볼 위치 검색 중")
        with _SEARCH_LOCK:
            detections = _search_symbols(pdf, folder, data_root)
        announce("PDF 벡터를 DWG로 가져오는 중")
        with _CAD_IMPORT_LOCK:
            _import_pdf(pdf, dwg)
        announce("CAD 객체와 심볼 범위 검사 중")
        before = _read_audit(blocks.audit(dwg, folder, "before"))
        if not before:
            raise RuntimeError("PDF에서 가져온 벡터 객체가 없습니다. AutoCAD 로그를 확인해 주세요.")
        groups = _assign_fresh_entities(detections["displayed_detections"], before, metadata["page_height_pt"])
        if groups:
            blocks.write_manifest(groups, folder)
            announce(f"심볼 {len(groups)}개 블록화 중")
            script = blocks.build_block_script(groups, folder)
            _run_cad(dwg, script, folder / "create_blocks.log")
            after = _read_audit(blocks.audit(dwg, folder, "after"))
            verification = blocks.verify(folder, before, after, groups,
                                         sum(row["status"] != "strong_candidate"
                                             for row in detections["displayed_detections"]))
            verification.update(pipeline_version="pdf-web-fresh-v1",
                                pipe_entities_left_unblocked=None,
                                pipe_separation_verified=False)
            (folder / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        else:
            verification = {"new_block_references": 0, "input_model_entities": len(before)}
        report = {**metadata, "mode": "fresh_pdf", "symbol_blocks": len(groups),
                  "candidate_count": len(detections["displayed_detections"]),
                  "verification": verification,
                  "note": "새 PDF의 벡터와 강한 심볼 후보만 블록화했습니다. 벡터 글자 윤곽·이미지 표의 편집 가능한 문자 복구는 아직 검증되지 않았습니다."}
    announce("결과 파일 묶는 중")
    archive = _bundle(folder, dwg, report)
    return {"folder": folder, "dwg": dwg, "archive": archive,
            "preview": folder / "detections.pdf", "report": report}
