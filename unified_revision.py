"""One-click local drawing comparison followed by optional LLM adjudication.

PDF/image bytes stay in this process.  The only object intended for an external
LLM is ``evidence``: OCR strings, coordinates, locally detected objects,
connection candidates, and raster-difference statistics.
"""
from __future__ import annotations

import time

import cv2

from llm_revision import build_revision_evidence
from revision_compare import compare_documents, serializable_report
from topology import build_topology


def _ocr_report(texts, width, height, dpi):
    return {"runs": {str(dpi): {
        "dpi": dpi,
        "image_size": [int(width), int(height)],
        "count": len(texts),
        "texts": texts,
    }}}


def _structure(gray, texts, objects, warnings):
    structure = {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "objects": objects,
        "texts": texts,
        "warnings": list(warnings),
    }
    try:
        structure["graph"] = build_topology(gray, objects, texts, max_segments=200)
    except (ValueError, cv2.error) as exc:
        structure["graph"] = {"nodes": [], "edges": [], "unresolved": [], "reviewed": False}
        structure["warnings"].append(f"배선 그래프 후보 생성 생략: {exc}")
    return structure


def _side_text(candidate, side):
    values = []
    for item in candidate[side]["ocr"]:
        value = str(item.get("text", "")).strip()
        if value and value not in values:
            values.append(value)
    return " | ".join(values) or None


def local_preliminary_review(evidence):
    """Produce useful review rows when no API key is supplied.

    These rows deliberately remain probable/uncertain.  They let the complete
    UI run offline without pretending that pixel arithmetic understands design
    intent.
    """
    changes = []
    for candidate in evidence["candidates"]:
        before_value = _side_text(candidate, "before")
        after_value = _side_text(candidate, "after")
        visual = candidate["visual"]
        added = int(visual.get("added_ink_pixels") or 0)
        removed = int(visual.get("removed_ink_pixels") or 0)
        if before_value != after_value and (before_value or after_value):
            if before_value and after_value:
                change_type, reason = "text_value_changed", "같은 변경 영역의 전후 OCR 문자열이 다릅니다."
            elif after_value:
                change_type, reason = "text_added", "후도면 변경 영역에서만 OCR 문자열이 검출됐습니다."
            else:
                change_type, reason = "text_removed", "전도면 변경 영역에서만 OCR 문자열이 검출됐습니다."
            verdict, confidence = "probable", .58
        else:
            if added > removed * 1.5:
                change_type, reason = "geometry_added", "후도면 잉크 증가가 우세하지만 설계 의미는 아직 판정하지 않았습니다."
            elif removed > added * 1.5:
                change_type, reason = "geometry_removed", "전도면 잉크 소실이 우세하지만 설계 의미는 아직 판정하지 않았습니다."
            else:
                change_type, reason = "unknown", "픽셀 변화는 있으나 OCR만으로 의미를 확정할 수 없습니다."
            verdict, confidence = "uncertain", .35
        changes.append({
            "change_id": candidate["change_id"],
            "verdict": verdict,
            "change_type": change_type,
            "before_value": before_value,
            "after_value": after_value,
            "affected_object_ids": [],
            "reason": reason,
            "confidence": confidence,
            "needs_review": True,
        })
    return {
        "summary": "로컬 후보 판정입니다. API 키를 넣고 LLM 분석을 켜면 OCR·기호·배선 근거를 함께 해석합니다.",
        "changes": changes,
        "global_warnings": ["로컬 판정은 설계 의미를 확정하지 않습니다."],
        "analysis_source": "local_preliminary",
    }


def run_local_analysis(before_data, after_data, before_type, after_type, *, dpi=100,
                       sensitivity="high", ocr_reader=None, ocr_confidence=.2,
                       detector=None, detector_confidence=.25, detector_imgsz=1024,
                       class_map=None):
    """Run every local stage and return one result for the Streamlit UI."""
    started = time.perf_counter()
    visual = compare_documents(
        before_data, after_data, before_type, after_type, dpi=dpi,
        ocr_reader=ocr_reader, min_confidence=ocr_confidence,
        sensitivity=sensitivity,
    )
    visual_seconds = time.perf_counter() - started
    before_gray = cv2.cvtColor(visual["before"], cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(visual["after_aligned"], cv2.COLOR_RGB2GRAY)

    texts_before, texts_after = [], []
    ocr_started = time.perf_counter()
    if ocr_reader is not None:
        from models_local import recognize_texts
        texts_before = recognize_texts(before_gray, ocr_reader, min_confidence=ocr_confidence)
        texts_after = recognize_texts(after_gray, ocr_reader, min_confidence=ocr_confidence)
    ocr_seconds = time.perf_counter() - ocr_started

    objects_before, objects_after = [], []
    detector_started = time.perf_counter()
    if detector is not None:
        from models_local import detect_symbols
        objects_before = detect_symbols(before_gray, detector, class_map=class_map,
                                        conf=detector_confidence, imgsz=detector_imgsz)
        objects_after = detect_symbols(after_gray, detector, class_map=class_map,
                                       conf=detector_confidence, imgsz=detector_imgsz)
    detector_seconds = time.perf_counter() - detector_started

    warnings = []
    if ocr_reader is None:
        warnings.append("PaddleOCR가 꺼져 있어 문자·숫자 변경은 픽셀 후보로만 남습니다.")
    if detector is None:
        warnings.append("검증된 YOLO 가중치가 없어 기호 종류 비교는 실행하지 않았습니다.")
    before_structure = _structure(before_gray, texts_before, objects_before, warnings)
    after_structure = _structure(after_gray, texts_after, objects_after, warnings)
    report = serializable_report(visual)
    before_ocr = _ocr_report(texts_before, before_gray.shape[1], before_gray.shape[0], dpi)
    after_ocr = _ocr_report(texts_after, after_gray.shape[1], after_gray.shape[0], dpi)
    evidence = build_revision_evidence(
        report, [before_gray.shape[1], before_gray.shape[0]],
        before_ocr_report=before_ocr, after_ocr_report=after_ocr,
        before_structure=before_structure, after_structure=after_structure,
    )
    return {
        "visual": visual,
        "report": report,
        "before_texts": texts_before,
        "after_texts": texts_after,
        "before_structure": before_structure,
        "after_structure": after_structure,
        "evidence": evidence,
        "local_review": local_preliminary_review(evidence),
        "timings": {
            "visual_seconds": visual_seconds,
            "ocr_seconds": ocr_seconds,
            "detector_seconds": detector_seconds,
            "total_local_seconds": time.perf_counter() - started,
        },
        "warnings": warnings,
    }
