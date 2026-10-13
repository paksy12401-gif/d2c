from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def _fallback_description(stats: dict[str, Any]) -> tuple[str, str, float]:
    added = int(stats["added_pixels"])
    deleted = int(stats["deleted_pixels"])
    added_length = float(stats.get("added_length", 0.0))
    deleted_length = float(stats.get("deleted_length", 0.0))
    width = float(stats.get("width", 0.0))
    height = float(stats.get("height", 0.0))
    added_components = int(stats.get("added_components", 0))
    deleted_components = int(stats.get("deleted_components", 0))
    added_loops = int(stats.get("added_closed_contours", 0))
    deleted_loops = int(stats.get("deleted_closed_contours", 0))
    shift = stats.get("estimated_local_shift")
    aligned_residual = stats.get("aligned_residual_ratio")
    location = f"약 {width:.1f} x {height:.1f} PDF 좌표 범위"
    structure = (
        f"추가 선 길이 약 {added_length:.1f}, 삭제 선 길이 약 {deleted_length:.1f}; "
        f"추가 묶음 {added_components}개, 삭제 묶음 {deleted_components}개"
    )
    if added_loops or deleted_loops:
        structure += f"; 추가 폐곡선 {added_loops}개, 삭제 폐곡선 {deleted_loops}개"

    if shift and aligned_residual is not None and added and deleted:
        dx = float(shift.get("dx", 0.0))
        dy = float(shift.get("dy", 0.0))
        residual = float(aligned_residual)
        if residual <= 0.14:
            return (
                "MOVED",
                f"{location}의 형상이 X {dx:+.2f}, Y {dy:+.2f}만큼 이동한 것으로 계산됩니다. "
                f"이동 보정 후 잔차는 {residual:.1%}입니다. {structure}.",
                0.86,
            )
        if residual <= 0.48:
            return (
                "MOVED_AND_MODIFIED",
                f"{location}의 형상이 X {dx:+.2f}, Y {dy:+.2f} 방향으로 재배치되면서 일부 구조도 변경되었습니다. "
                f"이동 보정 후에도 {residual:.1%}가 일치하지 않습니다. {structure}.",
                0.78,
            )
    if added and not deleted:
        return "ADDED", f"{location}에 기존 도면에는 없던 형상이 추가되었습니다. {structure}.", 0.78
    if deleted and not added:
        return "DELETED", f"{location}의 기존 형상이 새 도면에서 삭제되었습니다. {structure}.", 0.78
    if added > deleted * 1.8:
        return (
            "GEOMETRY_MODIFIED",
            f"{location}에서 기존 형상의 일부를 유지한 채 새로운 선과 윤곽이 주로 추가되었습니다. {structure}.",
            0.74,
        )
    if deleted > added * 1.8:
        return (
            "GEOMETRY_MODIFIED",
            f"{location}에서 기존 선과 윤곽이 주로 제거되었고 일부 형상은 새 구조로 교체되었습니다. {structure}.",
            0.74,
        )
    return (
        "GEOMETRY_MODIFIED",
        f"{location}에서 기존 형상의 삭제와 새로운 형상의 추가가 함께 발생했습니다. {structure}.",
        0.72,
    )


def describe_change(
    image_path: Path, stats: dict[str, Any], *, use_llm: bool = False
) -> tuple[str, str, float]:
    if not use_llm:
        return _fallback_description(stats)
    api_url = os.getenv("DRAWING_COMPARE_LLM_URL")
    api_key = os.getenv("DRAWING_COMPARE_LLM_API_KEY")
    model = os.getenv("DRAWING_COMPARE_LLM_MODEL")
    if not (api_url and api_key and model):
        return _fallback_description(stats)

    mime = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = (
        "이미지는 OLD, NEW, OVERLAY 세 패널로 된 CAD 변경 후보입니다. OVERLAY에서 회색은 공통, "
        "빨강은 기존 도면에만 존재, 초록은 새 도면에만 존재합니다. 제공된 수치에 근거해 "
        "어떤 구조가 추가/삭제/이동/교체됐는지 한국어로 구체적으로 설명하세요. "
        "보이지 않는 문자나 부품명을 추측하지 말고 판단 근거 수치를 포함하세요. "
        "JSON만 반환하세요: "
        '{"category":"GEOMETRY_MODIFIED|TEXT_CHANGED|DIMENSION_CHANGED|CONNECTION_CHANGED|ADDED|DELETED",'
        '"description":"...","confidence":0.0}. '
        f"측정값: {json.dumps(stats, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return (
            str(parsed.get("category", "GEOMETRY_MODIFIED")),
            str(parsed.get("description", "형상이 변경되었습니다.")),
            float(parsed.get("confidence", 0.5)),
        )
    except Exception:
        return _fallback_description(stats)
