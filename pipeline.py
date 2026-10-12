"""Local semantics. Recognition and relationship choices remain reviewable."""
import copy
import json
import math
import re
from pathlib import Path


def load_rules():
    return json.loads(Path(__file__).with_name("rules.json").read_text(encoding="utf-8"))


def validate_structure(data):
    if not isinstance(data, dict):
        raise ValueError("구조 JSON은 객체여야 합니다.")
    for key in ("width", "height"):
        value = data.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("양수 width/height가 필요합니다.")
    for key in ("objects", "texts"):
        items = data.get(key)
        if not isinstance(items, list) or len(items) > 10000:
            raise ValueError("objects/texts는 최대 10000개 배열이어야 합니다.")
        ids = set()
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("각 항목에 문자열 id가 필요합니다.")
            if not item["id"] or len(item["id"]) > 128 or item["id"] in ids:
                raise ValueError("id가 비어 있거나 중복되거나 너무 깁니다.")
            ids.add(item["id"])
    object_ids = {obj["id"] for obj in data["objects"]}
    kinds = {"line", "wire", "circle", "arc", "spline", "polyline", "connector", "terminal", "ground", "splice", "symbol"}
    for obj in data["objects"]:
        if obj.get("kind") not in kinds or not isinstance(obj.get("geometry"), dict):
            raise ValueError("허용된 객체 종류와 geometry가 필요합니다.")
    evidence_types = load_rules()["evidence_types"]
    for item in data["texts"]:
        if not isinstance(item.get("text"), str) or len(item["text"]) > 2048:
            raise ValueError("OCR text는 최대 2048자 문자열이어야 합니다.")
        bbox = item.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in bbox)
                or bbox[2] <= 0 or bbox[3] <= 0):
            raise ValueError("문자 bbox는 [x, y, 양수 폭, 양수 높이]여야 합니다.")
        links = item.get("links", [])
        if not isinstance(links, list) or len(links) > 32:
            raise ValueError("문자 관계 후보는 최대 32개입니다.")
        seen = set()
        for link in links:
            if not isinstance(link, dict) or link.get("object_id") not in object_ids:
                raise ValueError("관계 후보가 존재하지 않는 객체를 가리킵니다.")
            if link["object_id"] in seen or link.get("evidence") not in evidence_types:
                raise ValueError("중복 후보 또는 허용되지 않은 관계 근거입니다.")
            seen.add(link["object_id"])
    return data


def classify(text, rules):
    text = text.strip()
    for rule in rules["patterns"]:
        match = re.fullmatch(rule["pattern"], text, flags=re.IGNORECASE)
        if match:
            value = match["value"]
            if rule["token_kind"] != "identifier":
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("유효하지 않은 숫자입니다.")
            return {"roles": [rule["role"]], "value": value,
                    "unit": rule["unit"], "token_kind": rule["token_kind"]}
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("유효하지 않은 숫자입니다.")
        return {"roles": rules["bare_number_roles"], "value": value,
                "unit": "unknown", "token_kind": "bare_number"}
    return {"roles": [], "value": text, "unit": "none", "token_kind": "unknown"}


def analyze(structure, rules=None):
    validate_structure(structure)
    rules = rules or load_rules()
    objects = {obj["id"]: obj for obj in structure["objects"]}
    rows = []
    for item in structure["texts"]:
        row = {"text_id": item["id"], "raw_text": item["text"], **classify(item["text"], rules)}
        candidates = []
        for link in item.get("links", []):
            obj = objects[link["object_id"]]
            roles = [role for role in row["roles"] if obj["kind"] in rules["targets"][role]]
            if roles:
                candidates.append({"object_id": obj["id"], "kind": obj["kind"],
                                   "evidence": link["evidence"], "roles": roles})
        row["candidates"] = candidates
        rows.append(row)
    return rows


def suggest_links(structure, max_distance_px=40):
    """Local proximity candidates only; adjacency is never electrical proof."""
    validate_structure(structure)
    if type(max_distance_px) not in (int, float) or not math.isfinite(max_distance_px) or max_distance_px <= 0:
        raise ValueError("후보 거리 제한은 양수여야 합니다.")
    result, rules = copy.deepcopy(structure), load_rules()
    # ponytail: O(texts*objects) baseline; use a spatial index for dense drawings.
    for text in result["texts"]:
        if text.get("links"):
            continue
        roles = classify(text["text"], rules)["roles"]
        x, y, w, h = text["bbox"]
        p = (x+w/2, y+h/2)
        candidates = []
        for obj in result["objects"]:
            if not any(obj["kind"] in rules["targets"][role] for role in roles):
                continue
            geo = obj["geometry"]
            if "start" in geo and "end" in geo:
                a, b = geo["start"], geo["end"]
                dx, dy = b[0]-a[0], b[1]-a[1]
                length2 = dx*dx+dy*dy
                t = max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length2)) if length2 else 0
                distance = math.dist(p, (a[0]+t*dx, a[1]+t*dy))
            elif "center" in geo:
                distance = abs(math.dist(p, geo["center"])-geo.get("radius", 0))
            elif "insert" in geo:
                distance = math.dist(p, geo["insert"])
            else:
                continue
            if distance <= max_distance_px:
                candidates.append((distance, obj["id"]))
        text["links"] = [{"object_id": oid, "evidence": "adjacent"} for _, oid in sorted(candidates)[:4]]
    return result


def finalize(structure, decisions):
    """decisions: text id -> {role, object_id}; supplied only after local review.

    API suggestions are not accepted implicitly. Geometry is never rewritten
    from a guessed dimension or an LLM answer.
    """
    rules = load_rules()
    rows = analyze(structure, rules)
    if set(decisions) - {row["text_id"] for row in rows}:
        raise ValueError("알 수 없는 문자에 대한 결정입니다.")
    objects = {obj["id"]: {"object_id": obj["id"], "shape": obj["kind"],
                            "geometry_px": copy.deepcopy(obj["geometry"]),
                            "attributes": {}} for obj in structure["objects"]}
    unresolved = []
    for row in rows:
        decision = decisions.get(row["text_id"])
        if not decision:
            unresolved.append({"text_id": row["text_id"], "text": row["raw_text"],
                               "candidate_roles": row["roles"], "status": "needs_review"})
            continue
        role, target = decision.get("role"), decision.get("object_id")
        if (role not in row["roles"] or target not in objects
                or objects[target]["shape"] not in rules["targets"][role]):
            raise ValueError("문자 의미와 객체 종류가 맞지 않습니다.")
        value = row["value"]
        if role in ("length", "diameter", "wire_size") and value <= 0:
            raise ValueError("길이·직경·전선 단면적은 양수여야 합니다.")
        if role in ("pin_number", "quantity") and (value < 0 or not float(value).is_integer()):
            raise ValueError("핀 번호·수량은 음수가 아닌 정수여야 합니다.")
        unit = row["unit"]
        if row["token_kind"] == "bare_number":
            unit = rules["default_length_unit"] if role == "length" else "none"
        attrs = objects[target]["attributes"]
        if role in attrs:
            raise ValueError("같은 객체 속성이 중복됩니다. 로컬에서 충돌을 검토하세요.")
        attrs[role] = {"value": value, "unit": unit, "source_text_id": row["text_id"],
                       "verification": "user_confirmed"}
    return {"coordinate_system": "processed_image_pixels_y_down", "objects": list(objects.values()),
            "unresolved": unresolved, "status": "needs_review" if unresolved else "reviewed"}
