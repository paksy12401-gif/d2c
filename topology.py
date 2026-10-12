"""Local, reviewable wire candidates and revision comparison; no network calls.

Hough lines are a teaching baseline, not a skeleton model or a validated
electrical netlist. Crossings never create electrical connections automatically.
"""
from copy import deepcopy
from collections import Counter
from itertools import combinations
import math

import cv2
import numpy as np


def _point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("좌표는 유한한 숫자 [x, y]여야 합니다.")
    return list(value)


def _index(items, key):
    if not isinstance(items, (list, tuple)) or len(items) > 10000:
        raise ValueError("객체는 10,000개 이하의 목록이어야 합니다.")
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get(key), str) or not item[key]:
            raise ValueError(f"각 항목에 문자열 {key}가 필요합니다.")
        if item[key] in result:
            raise ValueError(f"중복 ID: {item[key]}")
        result[item[key]] = item
    return result


def _mapping(mapping, before, after):
    if (not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str)
            or k not in before or v not in after for k, v in mapping.items())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("대응표는 실제 ID 사이의 일대일 매핑이어야 합니다.")


def _graph(graph):
    if not isinstance(graph, dict):
        raise ValueError("graph는 JSON 객체여야 합니다.")
    nodes, edges = _index(graph.get("nodes", []), "id"), _index(graph.get("edges", []), "id")
    for node in nodes.values():
        _point(node["point"])
    for edge in edges.values():
        if edge.get("source") not in nodes or edge.get("target") not in nodes:
            raise ValueError("연결선이 존재하지 않는 노드를 참조합니다.")
    if graph.get("reviewed") is True and (any(e.get("status") not in ("confirmed", "rejected") for e in edges.values())
            or any(i.get("status") not in ("confirmed", "rejected") for i in graph.get("unresolved", []))):
        raise ValueError("후보/미해결 항목이 남은 그래프를 reviewed로 표시할 수 없습니다.")
    return nodes, edges


def build_topology(gray, objects=(), texts=(), snap_px=3.0, max_segments=200):
    """Keep every segment endpoint distinct; report proposed joins for review.

    Reuse objects with kind=line/wire and geometry.start/end when supplied.
    Otherwise Hough operates on a local binary image with OCR boxes masked out.
    All coordinates are processed-image pixels, x right / y down.
    """
    if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2
            or gray.size == 0 or gray.size > 20_000_000):
        raise ValueError("2,000만 픽셀 이하 uint8 흑백 이미지가 필요합니다.")
    if type(snap_px) not in (int, float) or not math.isfinite(snap_px) or snap_px <= 0:
        raise ValueError("snap_px는 양수여야 합니다.")
    if type(max_segments) is not int or not 1 <= max_segments <= 500:
        raise ValueError("max_segments는 1~500이어야 합니다.")
    _index(objects, "id")
    _index(texts, "id")
    segments = []
    for obj in objects:
        geo = obj.get("geometry", {})
        if not isinstance(geo, dict):
            raise ValueError("geometry는 JSON 객체여야 합니다.")
        if obj.get("kind") in ("line", "wire") and "start" in geo and "end" in geo:
            segments.append((_point(geo["start"]), _point(geo["end"])))
    warnings = ["Hough 후보: 중복·끊김·곡선 누락 가능. 중심선/전기 연결 정확도를 보장하지 않습니다.",
                "접속점, 비접속 교차, 커넥터 핀, bundle 분기와 splice는 로컬 검토가 필요합니다."]
    if not segments:
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        height, width = gray.shape
        for text in texts:
            box = text.get("bbox")
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                    or box[2] <= 0 or box[3] <= 0):
                raise ValueError("OCR bbox는 [x, y, 양수 폭, 양수 높이]여야 합니다.")
            x, y, w, h = box
            binary[max(0, int(y)):max(0, min(height, math.ceil(y + h))),
                   max(0, int(x)):max(0, min(width, math.ceil(x + w)))] = 0
        found = cv2.HoughLinesP(binary, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=3)
        if found is not None:
            segments = [(row[:2], row[2:]) for row in found.reshape(-1, 4).tolist()]
    if len(segments) > max_segments:
        warnings.append(f"{len(segments)}개 중 {max_segments}개 선분만 처리합니다. 전체 도면 결과가 아닙니다.")
    nodes, edges, unresolved = [], [], []
    for start, end in segments[:max_segments]:
        if start == end:
            continue
        a, b = f"N{len(nodes) + 1}", f"N{len(nodes) + 2}"
        nodes.extend([{"id": a, "point": start, "kind": "endpoint"},
                      {"id": b, "point": end, "kind": "endpoint"}])
        edges.append({"id": f"E{len(edges) + 1}", "source": a, "target": b,
                      "geometry": {"points": [start, end]}, "status": "candidate"})
    # ponytail: O(n²) capped at 500 segments; use a spatial index for larger drawings.
    parents = list(range(len(nodes)))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for i, j in combinations(range(len(nodes)), 2):
        if math.dist(nodes[i]["point"], nodes[j]["point"]) <= snap_px:
            parents[root(j)] = root(i)
    groups = {}
    for i, node in enumerate(nodes):
        groups.setdefault(root(i), []).append(node["id"])
    for group in groups.values():
        if len(group) > 1:
            unresolved.append({"kind": "endpoint_cluster", "node_ids": group, "status": "needs_review"})
    for first, second in combinations(edges, 2):
        p, q = np.array(first["geometry"]["points"], dtype=float), np.array(second["geometry"]["points"], dtype=float)
        r, s = p[1] - p[0], q[1] - q[0]
        cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
        denominator = cross(r, s)
        if abs(denominator) < 1e-9:
            continue
        t, u = cross(q[0] - p[0], s) / denominator, cross(q[0] - p[0], r) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1 and (0 < t < 1 or 0 < u < 1):
            unresolved.append({"kind": "crossing", "edge_ids": [first["id"], second["id"]],
                               "point": (p[0] + t * r).tolist(), "connectivity": "unknown", "status": "needs_review"})
    for obj in objects:
        if obj.get("kind") in ("connector", "terminal", "ground", "splice", "symbol", "connector_pin"):
            unresolved.append({"kind": "pin_or_symbol_association", "object_id": obj["id"], "status": "needs_review"})
    return {"coordinate_system": "processed_image_pixels_y_down", "nodes": nodes, "edges": edges,
            "unresolved": unresolved, "warnings": warnings, "reviewed": False}


def confirm_edges(graph, edge_ids):
    """Human approval of segments only; does not resolve crossings or join nodes."""
    result = deepcopy(graph)
    _, edges = _graph(result)
    if not isinstance(edge_ids, (list, tuple, set)) or set(edge_ids) - edges.keys():
        raise ValueError("승인할 실제 선분 ID 목록이 필요합니다.")
    for key in edge_ids:
        edges[key]["status"] = "confirmed"
    return result


def connect_nodes(graph, node_ids):
    """Explicit human-confirmed endpoint join. Interior crossings need split edges first."""
    result = deepcopy(graph)
    nodes, edges = _graph(result)
    if not isinstance(node_ids, (list, tuple)) or len(set(node_ids)) < 2 or set(node_ids) - nodes.keys():
        raise ValueError("접속을 확정할 서로 다른 실제 노드 ID 두 개 이상이 필요합니다.")
    selected, keep = set(node_ids), node_ids[0]
    if any({edge["source"], edge["target"]} <= selected for edge in edges.values()):
        raise ValueError("한 선분의 양끝을 합칠 수 없습니다. 닫힌 경로는 별도 검토하세요.")
    for edge in edges.values():
        for key in ("source", "target"):
            if edge[key] in selected:
                edge[key] = keep
    nodes[keep].update(kind="user_confirmed_junction", merged_node_ids=sorted(selected))
    result["nodes"] = [node for key, node in nodes.items() if key == keep or key not in selected]
    for issue in result.get("unresolved", []):
        if issue.get("kind") == "endpoint_cluster" and set(issue["node_ids"]) <= selected:
            issue["status"] = "confirmed"
        if "node_ids" in issue:
            issue["node_ids"] = sorted({keep if key in selected else key for key in issue["node_ids"]})
    result["reviewed"] = False  # Other crossings, gaps and pin associations remain open.
    return result


def compare_revisions(before, after, matches, *, node_matches=None, matches_complete=False,
                      registration=None, mm_per_px=None):
    """Compare finalized objects using explicitly supplied one-to-one matches.

    registration={matrix: 2x3 affine from AFTER pixels to BEFORE, verified: True}.
    Geometry comparison only supports straight segments or equal-length point
    lists. mm_per_px is the calibrated BEFORE raster scale, not PDF render dpi.
    Unknown matching/recognition is never converted to a confirmed deletion.
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("전후 결과는 JSON 객체여야 합니다.")
    old, new = _index(before.get("objects", []), "object_id"), _index(after.get("objects", []), "object_id")
    _mapping(matches, old, new)
    if type(matches_complete) is not bool:
        raise ValueError("matches_complete는 명시적인 bool이어야 합니다.")
    changes, notes = [], []
    status = "confirmed" if matches_complete and before.get("status") == after.get("status") == "reviewed" else "candidate"
    for key in old.keys() - matches.keys():
        changes.append({"kind": "object_removed", "before_id": key, "status": status})
    for key in new.keys() - set(matches.values()):
        changes.append({"kind": "object_added", "after_id": key, "status": status})
    matrix = None
    if registration is not None:
        if not isinstance(registration, dict) or registration.get("verified") is not True:
            raise ValueError("검증된 registration 설정이 필요합니다.")
        matrix = np.asarray(registration.get("matrix"), dtype=float)
        if (matrix.shape != (2, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-12
                or type(mm_per_px) not in (int, float) or not math.isfinite(mm_per_px) or mm_per_px <= 0):
            raise ValueError("가역 2x3 정렬행렬과 양수 mm_per_px가 필요합니다.")
    for left, right in matches.items():
        a, b = old[left], new[right]
        attrs_a, attrs_b = a.get("attributes", {}), b.get("attributes", {})
        if not isinstance(attrs_a, dict) or not isinstance(attrs_b, dict):
            raise ValueError("attributes는 JSON 객체여야 합니다.")
        for key in attrs_a.keys() | attrs_b.keys():
            av, bv = attrs_a.get(key), attrs_b.get(key)
            value = lambda item: (item.get("value"), item.get("unit")) if isinstance(item, dict) else item
            if value(av) != value(bv):
                verified = all(isinstance(v, dict) and v.get("verification") == "user_confirmed" for v in (av, bv))
                changes.append({"kind": "attribute_changed", "before_id": left, "after_id": right,
                                "attribute": key, "before": av, "after": bv,
                                "status": "confirmed" if verified else "candidate"})
        if a.get("shape") != b.get("shape"):
            changes.append({"kind": "object_type_changed", "before_id": left, "after_id": right, "status": "candidate"})
        if matrix is not None:
            def points(obj):
                geo = obj.get("geometry_px", {})
                raw = [geo["start"], geo["end"]] if "start" in geo and "end" in geo else geo.get("points", [])
                return np.array([_point(p) for p in raw], dtype=float)
            p, q = points(a), points(b)
            if len(p) and p.shape == q.shape:
                q = np.column_stack((q, np.ones(len(q)))) @ matrix.T
                delta = min(np.linalg.norm(p - q, axis=1).max(), np.linalg.norm(p - q[::-1], axis=1).max()) * mm_per_px
                if delta > 1e-6:
                    changes.append({"kind": "geometry_changed", "before_id": left, "after_id": right,
                                    "max_point_displacement_mm": float(delta), "status": "candidate"})
            else:
                notes.append(f"{left}: 해당 기하의 정렬 비교는 미지원입니다.")
    topology_status = "unavailable"
    if node_matches is not None and "graph" in before and "graph" in after:
        ga, gb = before.get("graph", {}), after.get("graph", {})
        na, ea = _graph(ga)
        nb, eb = _graph(gb)
        _mapping(node_matches, na, nb)
        inverse = {v: k for k, v in node_matches.items()}
        def selected(edges, mapping):
            return Counter(tuple(sorted((mapping[e["source"]], mapping[e["target"]]))) for e in edges.values()
                           if e.get("status") == "confirmed" and e["source"] in mapping and e["target"] in mapping)
        left_edges, right_edges = selected(ea, node_matches), selected(eb, {key: key for key in inverse})
        topology_status = "compared" if len(node_matches) == len(na) == len(nb) else "partial"
        edge_status = "confirmed" if topology_status == "compared" and ga.get("reviewed") is True and gb.get("reviewed") is True else "candidate"
        for kind, difference in (("connection_removed", left_edges - right_edges), ("connection_added", right_edges - left_edges)):
            changes.extend({"kind": kind, "after_node_ids": list(pair), "count": count, "status": edge_status}
                           for pair, count in sorted(difference.items()))
    else:
        notes.append("전후 graph 또는 node_matches가 없어 연결 구조를 비교하지 않았습니다.")
    if matrix is None:
        notes.append("검증된 정렬·축척이 없어 기하/경로 길이를 비교하지 않았습니다.")
    return {"changes": changes, "topology_status": topology_status,
            "geometry_status": "compared_supported_paths" if matrix is not None else "unavailable", "notes": notes}


def self_check():
    image = np.full((100, 100), 255, np.uint8)
    cv2.line(image, (10, 50), (90, 50), 0, 2)
    assert build_topology(image)["edges"]
    objects = [{"id": "L1", "kind": "wire", "geometry": {"start": [10, 50], "end": [90, 50]}},
               {"id": "L2", "kind": "wire", "geometry": {"start": [50, 10], "end": [50, 90]}}]
    graph = build_topology(image, objects)
    assert len(graph["nodes"]) == 4 and len(graph["edges"]) == 2
    assert any(issue["kind"] == "crossing" for issue in graph["unresolved"])
    assert confirm_edges(graph, ["E1"])["edges"][0]["status"] == "confirmed"
    before = {"objects": [{"object_id": "A", "shape": "wire", "attributes": {"length": {
        "value": 150, "unit": "mm", "verification": "user_confirmed"}}}], "graph": graph, "status": "reviewed"}
    after = deepcopy(before)
    after["objects"][0]["object_id"] = "B"
    after["objects"][0]["attributes"]["length"]["value"] = 180
    result = compare_revisions(before, after, {"A": "B"}, matches_complete=True)
    assert result["changes"][0]["kind"] == "attribute_changed" and result["topology_status"] == "unavailable"
    assert all(change["status"] == "candidate" for change in compare_revisions(before, after, {})["changes"])
    before["objects"][0]["geometry_px"] = {"start": [0, 0], "end": [100, 0]}
    after["objects"][0]["geometry_px"] = {"start": [10, 0], "end": [110, 0]}
    aligned = compare_revisions(before, after, {"A": "B"}, mm_per_px=0.1,
                                registration={"verified": True, "matrix": [[1, 0, -10], [0, 1, 0]]})
    assert not any(c["kind"] == "geometry_changed" for c in aligned["changes"])
    reviewed = confirm_edges(graph, ["E1", "E2"])
    for issue in reviewed["unresolved"]:
        issue["status"] = "rejected"  # Synthetic crossing explicitly declared nonconnecting.
    reviewed["reviewed"] = True
    before["graph"], after["graph"] = reviewed, deepcopy(reviewed)
    node_matches = {node["id"]: "NEW_" + node["id"] for node in reviewed["nodes"]}
    for node in after["graph"]["nodes"]:
        node["id"] = node_matches[node["id"]]
    for edge in after["graph"]["edges"]:
        edge["source"], edge["target"] = node_matches[edge["source"]], node_matches[edge["target"]]
    after["graph"]["edges"].pop()
    compared = compare_revisions(before, after, {"A": "B"}, node_matches=node_matches)
    assert any(c["kind"] == "connection_removed" and c["status"] == "confirmed" for c in compared["changes"])
    joined = connect_nodes(graph, ["N2", "N3"])
    assert len(joined["nodes"]) == 3 and joined["edges"][1]["source"] == "N2"
    try:
        compare_revisions(before, after, {"MISSING": "B"})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown mapping accepted")
    print("topology synthetic checks passed")


if __name__ == "__main__":
    self_check()
