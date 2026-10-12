"""The only external HTTP boundary. Never accepts a drawing or raw OCR text."""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROLES = {"diameter", "voltage", "wire_size", "length", "connector_name", "pin_number", "quantity"}
KINDS = {"line", "wire", "circle", "arc", "spline", "connector", "terminal", "ground", "splice", "symbol"}
TOKENS = {"diameter_mark", "voltage_unit", "area_unit", "length_unit", "bare_number"}
UNITS = {"mm", "V", "mm2", "none", "unknown"}
EVIDENCE = {"leader", "dimension_line", "adjacent", "enclosed"}
ALIASES = ("a", "b", "c", "d")


def validate_payload(payload):
    """An allowlist, not a denylist: unexpected fields always fail closed."""
    if not isinstance(payload, dict) or set(payload) != {"token_kind", "unit", "options"}:
        raise ValueError("허용되지 않은 API 요청 필드입니다.")
    if (not isinstance(payload["token_kind"], str) or payload["token_kind"] not in TOKENS
            or not isinstance(payload["unit"], str) or payload["unit"] not in UNITS):
        raise ValueError("허용되지 않은 토큰/단위입니다.")
    options = payload["options"]
    if not isinstance(options, list) or not 1 <= len(options) <= 4:
        raise ValueError("API에는 관계 후보 1~4개만 전달할 수 있습니다.")
    for i, option in enumerate(options):
        if not isinstance(option, dict) or set(option) != {"target", "kind", "evidence", "roles"}:
            raise ValueError("허용되지 않은 후보 필드입니다.")
        if (option["target"] != ALIASES[i] or not isinstance(option["kind"], str)
                or option["kind"] not in KINDS or not isinstance(option["evidence"], str)
                or option["evidence"] not in EVIDENCE):
            raise ValueError("허용되지 않은 관계 후보입니다.")
        roles = option["roles"]
        if (not isinstance(roles, list) or not roles or len(roles) > len(ROLES)
                or any(not isinstance(role, str) or role not in ROLES for role in roles)
                or len(set(roles)) != len(roles)):
            raise ValueError("허용되지 않은 의미 후보입니다.")
    return payload


def make_payload(row):
    # Only controlled categories cross this boundary. Values/IDs stay in mapping.
    mapping = {}
    options = []
    if len(row["candidates"]) > 4:
        raise ValueError("후보가 너무 많습니다. 먼저 로컬에서 관계를 좁히세요.")
    for alias, candidate in zip(ALIASES, row["candidates"]):
        mapping[alias] = candidate["object_id"]
        options.append({"target": alias, "kind": candidate["kind"],
                        "evidence": candidate["evidence"], "roles": list(candidate["roles"])})
    payload = {"token_kind": row["token_kind"], "unit": row["unit"], "options": options}
    return validate_payload(payload), mapping


def response_schema(payload):
    roles = sorted({role for option in payload["options"] for role in option["roles"]})
    return {"type": "object", "properties": {
        "role": {"type": "string", "enum": roles + ["unknown"]},
        "target": {"type": ["string", "null"], "enum": [o["target"] for o in payload["options"]] + [None]}
    }, "required": ["role", "target"], "additionalProperties": False}


def validate_selection(selection, payload):
    validate_payload(payload)
    if not isinstance(selection, dict) or set(selection) != {"role", "target"}:
        raise ValueError("응답 필드가 올바르지 않습니다.")
    if selection == {"role": "unknown", "target": None}:
        return selection
    if not any(selection["target"] == o["target"] and selection["role"] in o["roles"]
               for o in payload["options"]):
        raise ValueError("API가 입력 후보에 없는 의미 또는 객체를 반환했습니다.")
    return selection


def request_body(payload, model):
    validate_payload(payload)
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", model):
        raise ValueError("유효한 API 모델 ID가 필요합니다.")
    return {"model": model, "store": False,
            "instructions": (
                "Select a semantic relation from the supplied categorical options. "
                "You have no drawing, OCR text, actual values or coordinates. "
                "Do not invent missing context. Adjacency alone is not proof. "
                "If evidence is insufficient or ambiguous, return role unknown and target null. "
                "All selections require local human review."),
            "input": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            "text": {"format": {"type": "json_schema", "name": "relation_selection",
                                "strict": True, "schema": response_schema(payload)}},
            "max_output_tokens": 2048}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("API 리다이렉트는 허용하지 않습니다.")


def request_selection(payload, model, *, enabled=False, api_key=None):
    if enabled is not True:
        raise PermissionError("외부 API 전송이 비활성화되어 있습니다.")
    body = request_body(payload, model)
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("화면에 API 키를 입력하거나 로컬 환경변수 OPENAI_API_KEY를 설정하세요.")
    if not isinstance(key, str) or any(c.isspace() for c in key):
        raise ValueError("API 키에 공백이나 줄바꿈이 포함되어 있습니다.")
    request = Request("https://api.openai.com/v1/responses",
                      data=json.dumps(body, ensure_ascii=True, allow_nan=False).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=45) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("API 응답이 허용 크기를 초과했습니다.")
        result = json.loads(raw)
    except HTTPError as exc:
        raise ValueError(f"API HTTP 오류 {exc.code}. 원본/응답 본문은 로그로 출력하지 않습니다.") from None
    except (URLError, TimeoutError):
        raise ValueError("API 연결 실패. 로컬 검토를 계속할 수 있습니다.") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("API 응답이 JSON 형식이 아닙니다.") from None
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("API 응답이 완료되지 않았습니다.")
    output = result.get("output")
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise ValueError("API output 형식이 올바르지 않습니다.")
    parts = []
    for item in output:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if (not isinstance(content, list) or any(not isinstance(part, dict) for part in content)
                or item.get("status", "completed") != "completed"):
            raise ValueError("API message 형식이 올바르지 않거나 완료되지 않았습니다.")
        parts.extend(content)
    if any(p.get("type") == "refusal" for p in parts):
        raise ValueError("API가 해석을 거절했습니다.")
    texts = [p.get("text") for p in parts if p.get("type") == "output_text"]
    if len(texts) != 1 or not isinstance(texts[0], str):
        raise ValueError("API에 단일 JSON 응답이 없습니다.")
    try:
        selection = json.loads(texts[0])
    except json.JSONDecodeError:
        raise ValueError("API가 유효한 선택 JSON을 반환하지 않았습니다.") from None
    return validate_selection(selection, payload)
