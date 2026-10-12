"""Rich structured LLM review for locally detected drawing changes.

Original PDF/image bytes never enter this module.  Unlike the legacy minimal
relation API, this boundary intentionally includes actual OCR strings,
coordinates, visual-difference measurements, nearby objects and graph evidence
so the model can actively adjudicate revision candidates.
"""
from __future__ import annotations

import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


VERDICTS = ("confirmed", "probable", "uncertain", "rendering_noise")
CHANGE_TYPES = (
    "text_value_changed", "text_added", "text_removed", "text_unreadable_changed",
    "symbol_added", "symbol_removed", "symbol_changed", "geometry_added",
    "geometry_removed", "geometry_moved", "connection_added", "connection_removed",
    "mixed_change", "no_semantic_change", "unknown",
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("API 리다이렉트는 허용하지 않습니다.")


def _finite_json(value, *, depth=0):
    if depth > 12:
        raise ValueError("LLM 변경 JSON의 중첩이 너무 깊습니다.")
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("LLM 변경 JSON에는 유한한 숫자만 허용합니다.")
        return
    if isinstance(value, list):
        if len(value) > 20000:
            raise ValueError("LLM 변경 JSON 배열이 너무 큽니다.")
        for item in value:
            _finite_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 500:
            raise ValueError("LLM 변경 JSON 객체가 너무 큽니다.")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("LLM 변경 JSON 키가 올바르지 않습니다.")
            _finite_json(item, depth=depth + 1)
        return
    raise ValueError("LLM 변경 JSON에 직렬화할 수 없는 값이 있습니다.")


def validate_evidence(payload):
    required = {"document", "alignment", "candidate_generation", "candidates"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("풍부한 변경 근거 JSON의 필드가 올바르지 않습니다.")
    _finite_json(payload)
    document = payload["document"]
    if (not isinstance(document, dict)
            or not isinstance(document.get("drawing_type"), str)
            or not isinstance(document.get("page_size_px"), list)
            or len(document["page_size_px"]) != 2
            or any(type(value) not in (int, float) or value <= 0 for value in document["page_size_px"])):
        raise ValueError("도면 종류와 페이지 크기가 필요합니다.")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or not 0 <= len(candidates) <= 100:
        raise ValueError("변경 후보는 0~100개여야 합니다.")
    ids = set()
    for item in candidates:
        keys = {"change_id", "bbox_px", "visual", "before", "after", "nearby_objects", "connections"}
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("변경 후보 필드가 올바르지 않습니다.")
        change_id = item["change_id"]
        if (not isinstance(change_id, str) or not re.fullmatch(r"CHANGE_[0-9]{3,6}", change_id)
                or change_id in ids):
            raise ValueError("변경 후보 ID가 잘못됐거나 중복됐습니다.")
        ids.add(change_id)
        box = item["bbox_px"]
        if (not isinstance(box, list) or len(box) != 4
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in box)
                or box[2] <= 0 or box[3] <= 0):
            raise ValueError("변경 후보 bbox가 올바르지 않습니다.")
        for side in ("before", "after"):
            evidence = item[side]
            if not isinstance(evidence, dict) or set(evidence) != {"ocr", "objects"}:
                raise ValueError("전후 OCR/객체 근거가 필요합니다.")
            if not isinstance(evidence["ocr"], list) or not isinstance(evidence["objects"], list):
                raise ValueError("OCR/객체 근거는 배열이어야 합니다.")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > 2_000_000:
        raise ValueError("변경 근거 JSON은 2MB 이하여야 합니다.")
    return payload


def _centre(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _inside(point, box, padding=0):
    x, y, width, height = box
    return x - padding <= point[0] <= x + width + padding and y - padding <= point[1] <= y + height + padding


def _scaled_ocr(texts, source_size, page_size):
    if not texts:
        return []
    sx, sy = page_size[0] / source_size[0], page_size[1] / source_size[1]
    output = []
    for item in texts:
        box = item.get("bbox", [])
        if len(box) != 4:
            continue
        output.append({
            "text": item.get("text", ""),
            "confidence": item.get("confidence"),
            "bbox_px": [float(box[0] * sx), float(box[1] * sy),
                        float(box[2] * sx), float(box[3] * sy)],
        })
    return output


def _nearby_ocr(texts, box):
    padding = max(20, round(max(box[2], box[3]) * .12))
    return [item for item in texts if _inside(_centre(item["bbox_px"]), box, padding)][:80]


def _object_point(obj):
    if isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4:
        return _centre(obj["bbox"])
    geometry = obj.get("geometry", {})
    for key in ("insert", "center", "start"):
        value = geometry.get(key)
        if isinstance(value, list) and len(value) == 2:
            return value
    return None


def _nearby_objects(structure, box):
    if not isinstance(structure, dict):
        return []
    result = []
    padding = max(30, round(max(box[2], box[3]) * .2))
    for obj in structure.get("objects", []):
        point = _object_point(obj)
        if point is not None and _inside(point, box, padding):
            result.append({key: obj.get(key) for key in
                           ("id", "kind", "class_name", "confidence", "bbox", "geometry")
                           if key in obj})
    return result[:100]


def _nearby_connections(structure, box):
    if not isinstance(structure, dict) or not isinstance(structure.get("graph"), dict):
        return []
    nodes = {node.get("id"): node for node in structure["graph"].get("nodes", [])}
    selected = {node_id for node_id, node in nodes.items()
                if isinstance(node.get("point"), list) and _inside(node["point"], box, 30)}
    return [{key: edge.get(key) for key in ("id", "source", "target", "status")}
            for edge in structure["graph"].get("edges", [])
            if edge.get("source") in selected or edge.get("target") in selected][:200]


def build_revision_evidence(report, page_size_px, *, before_ocr_report=None,
                            after_ocr_report=None, before_structure=None,
                            after_structure=None, drawing_type="electrical_wire_harness"):
    """Combine full local evidence for every visual change candidate."""
    if not isinstance(report, dict) or not isinstance(report.get("changes"), list):
        raise ValueError("직렬화된 변경 비교 보고서가 필요합니다.")
    page_size = [int(page_size_px[0]), int(page_size_px[1])]

    def ocr_items(audit):
        if not isinstance(audit, dict) or not isinstance(audit.get("runs"), dict) or not audit["runs"]:
            return []
        run = audit["runs"][sorted(audit["runs"], key=lambda value: int(value))[-1]]
        return _scaled_ocr(run.get("texts", []), run.get("image_size", page_size), page_size)

    before_texts, after_texts = ocr_items(before_ocr_report), ocr_items(after_ocr_report)
    candidates = []
    for change in report["changes"]:
        box = change["bbox_px"]
        before_objects = _nearby_objects(before_structure, box)
        after_objects = _nearby_objects(after_structure, box)
        candidates.append({
            "change_id": change["id"],
            "bbox_px": box,
            "visual": {
                "local_label": change.get("label"),
                "source": change.get("source"),
                "added_ink_pixels": change.get("added_ink_pixels"),
                "removed_ink_pixels": change.get("removed_ink_pixels"),
                "annotation_ocr_confidence": change.get("ocr_confidence"),
            },
            "before": {"ocr": _nearby_ocr(before_texts, box), "objects": before_objects},
            "after": {"ocr": _nearby_ocr(after_texts, box), "objects": after_objects},
            "nearby_objects": {"before_count": len(before_objects), "after_count": len(after_objects)},
            "connections": {"before": _nearby_connections(before_structure, box),
                            "after": _nearby_connections(after_structure, box)},
        })
    payload = {
        "document": {"drawing_type": drawing_type, "page_size_px": page_size,
                     "coordinate_system": "image_pixels_x_right_y_down"},
        "alignment": report.get("alignment", {}),
        "candidate_generation": {"method": "high_recall_raster_difference",
                                 "statistics": report.get("difference_stats")},
        "candidates": candidates,
    }
    return validate_evidence(payload)


def response_schema(change_ids):
    item = {"type": "object", "properties": {
        "change_id": {"type": "string", "enum": change_ids},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "change_type": {"type": "string", "enum": list(CHANGE_TYPES)},
        "before_value": {"type": ["string", "null"]},
        "after_value": {"type": ["string", "null"]},
        "affected_object_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
        "reason": {"type": "string", "maxLength": 1200},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "needs_review": {"type": "boolean"},
    }, "required": ["change_id", "verdict", "change_type", "before_value", "after_value",
                     "affected_object_ids", "reason", "confidence", "needs_review"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "summary": {"type": "string", "maxLength": 4000},
        "changes": {"type": "array", "items": item, "minItems": len(change_ids),
                    "maxItems": len(change_ids)},
        "global_warnings": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
    }, "required": ["summary", "changes", "global_warnings"], "additionalProperties": False}


def request_body(payload, model, *, draft=None):
    validate_evidence(payload)
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", model):
        raise ValueError("유효한 API 모델 ID가 필요합니다.")
    ids = [item["change_id"] for item in payload["candidates"]]
    instructions = (
        "You are the semantic revision analyst for an electrical wire-harness drawing. "
        "The input contains rich local evidence but no original drawing image. Compare before and after OCR, "
        "objects, geometry, connections, alignment quality, and ink counts. OCR may be wrong. A visual change "
        "without readable OCR must remain text_unreadable_changed or another evidence-backed type; never invent "
        "a value. Mark edge-only antialiasing or registration residue as rendering_noise only when the supplied "
        "evidence supports it. Return exactly one result for every change_id. Explain conclusions in Korean."
    )
    content = {"evidence": payload}
    if draft is not None:
        content["draft_for_independent_verification"] = draft
        instructions += " Independently verify the draft, correct unsupported conclusions, and return the final review."
    return {"model": model, "store": False, "instructions": instructions,
            "input": json.dumps(content, ensure_ascii=False, separators=(",", ":")),
            "text": {"format": {"type": "json_schema", "name": "drawing_revision_review",
                                "strict": True, "schema": response_schema(ids)}},
            "max_output_tokens": min(16000, 1200 + 450 * len(ids))}


def validate_review(review, payload):
    ids = [item["change_id"] for item in validate_evidence(payload)["candidates"]]
    if not isinstance(review, dict) or set(review) != {"summary", "changes", "global_warnings"}:
        raise ValueError("LLM 변경 검토 응답 필드가 올바르지 않습니다.")
    if (not isinstance(review["summary"], str) or not isinstance(review["changes"], list)
            or not isinstance(review["global_warnings"], list)):
        raise ValueError("LLM 변경 검토 응답 형식이 올바르지 않습니다.")
    if [item.get("change_id") for item in review["changes"]] != ids:
        raise ValueError("LLM은 입력 순서대로 모든 변경 후보를 한 번씩 반환해야 합니다.")
    expected = {"change_id", "verdict", "change_type", "before_value", "after_value",
                "affected_object_ids", "reason", "confidence", "needs_review"}
    for item in review["changes"]:
        if (not isinstance(item, dict) or set(item) != expected
                or item["verdict"] not in VERDICTS or item["change_type"] not in CHANGE_TYPES
                or type(item["confidence"]) not in (int, float)
                or not 0 <= item["confidence"] <= 1
                or type(item["needs_review"]) is not bool):
            raise ValueError("LLM 변경 후보 판정값이 올바르지 않습니다.")
    return review


def _extract_response(result):
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("API 응답이 완료되지 않았습니다.")
    parts = []
    for item in result.get("output", []):
        if isinstance(item, dict) and item.get("type") == "message":
            parts.extend(item.get("content", []))
    if any(isinstance(part, dict) and part.get("type") == "refusal" for part in parts):
        raise ValueError("API가 변경 해석을 거절했습니다.")
    texts = [part.get("text") for part in parts
             if isinstance(part, dict) and part.get("type") == "output_text"]
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("API에 단일 구조화 JSON 응답이 없습니다.")
    try:
        return json.loads(texts[0])
    except json.JSONDecodeError:
        raise ValueError("API 응답 JSON을 읽을 수 없습니다.") from None


def _call(body, api_key):
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key or not isinstance(key, str) or any(character.isspace() for character in key):
        raise ValueError("유효한 OpenAI API 키가 필요합니다.")
    request = Request("https://api.openai.com/v1/responses",
                      data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=120) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("API 응답이 4MB를 초과했습니다.")
        return json.loads(raw)
    except HTTPError as exc:
        raise ValueError(f"API HTTP 오류 {exc.code}. 응답 본문은 출력하지 않습니다.") from None
    except (URLError, TimeoutError):
        raise ValueError("API 연결 실패. 로컬 변경 후보는 유지됩니다.") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("API 응답이 JSON 형식이 아닙니다.") from None


def _review_batch(payload, model, api_key, passes):
    first = validate_review(_extract_response(_call(request_body(payload, model), api_key)), payload)
    if passes == 1:
        return first
    return validate_review(
        _extract_response(_call(request_body(payload, model, draft=first), api_key)), payload)


def request_revision_review(payload, model, *, enabled=False, api_key=None, passes=2,
                            batch_size=20):
    if enabled is not True:
        raise PermissionError("풍부한 변경 JSON의 외부 전송이 비활성화되어 있습니다.")
    if passes not in (1, 2):
        raise ValueError("LLM 검토 횟수는 1회 또는 2회여야 합니다.")
    payload = validate_evidence(payload)
    if not payload["candidates"]:
        raise ValueError("LLM이 판정할 변경 후보가 없습니다.")
    if type(batch_size) is not int or not 1 <= batch_size <= 30:
        raise ValueError("LLM 배치 크기는 1~30이어야 합니다.")
    candidates = payload["candidates"]
    reviews = []
    for offset in range(0, len(candidates), batch_size):
        batch = {**payload, "candidates": candidates[offset:offset + batch_size]}
        reviews.append(_review_batch(batch, model, api_key, passes))
    if len(reviews) == 1:
        result = reviews[0]
    else:
        warnings = []
        for review in reviews:
            for warning in review["global_warnings"]:
                if warning not in warnings:
                    warnings.append(warning)
        result = {
            "summary": f"{len(candidates)}개 후보를 {len(reviews)}개 배치로 판정했습니다. "
                       + " ".join(review["summary"] for review in reviews),
            "changes": [item for review in reviews for item in review["changes"]],
            "global_warnings": warnings,
        }
        validate_review(result, payload)
    result["analysis_source"] = "openai_llm"
    result["batch_count"] = len(reviews)
    result["passes_per_batch"] = passes
    return result
