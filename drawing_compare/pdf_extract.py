from __future__ import annotations

import math
from pathlib import Path

import pymupdf

from .models import Drawing, Segment, TextItem


def _point(value: object) -> tuple[float, float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise TypeError(f"지원하지 않는 PDF 좌표입니다: {value!r}")


def _segment(
    first: tuple[float, float],
    second: tuple[float, float],
    page_height: float,
    page_offset: float,
    kind: str,
    handle: str,
) -> Segment:
    return Segment(
        first[0],
        page_offset + page_height - first[1],
        second[0],
        page_offset + page_height - second[1],
        kind,
        handle,
    )


def _cubic_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    flattening: float,
) -> list[tuple[float, float]]:
    control_length = sum(math.dist(a, b) for a, b in ((p0, p1), (p1, p2), (p2, p3)))
    steps = max(4, min(96, int(math.ceil(control_length / max(flattening, 0.25)))))
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _append_polyline(
    target: list[Segment],
    points: list[tuple[float, float]],
    page_height: float,
    page_offset: float,
    kind: str,
    handle: str,
    close: bool = False,
) -> None:
    if close and points and points[0] != points[-1]:
        points = [*points, points[0]]
    for first, second in zip(points, points[1:]):
        segment = _segment(first, second, page_height, page_offset, kind, handle)
        if segment.length > 1e-7:
            target.append(segment)


def load_pdf(path: Path, flattening: float = 0.75) -> Drawing:
    document = pymupdf.open(path)
    if document.page_count == 0:
        raise ValueError(f"페이지가 없는 PDF입니다: {path}")

    segments: list[Segment] = []
    texts: list[TextItem] = []
    page_offset = 0.0
    try:
        for page_index, page in enumerate(document):
            page_height = float(page.rect.height)
            page_width = float(page.rect.width)
            for drawing_index, drawing in enumerate(page.get_drawings()):
                handle = f"P{page_index + 1}-D{drawing_index + 1}"
                path_start: tuple[float, float] | None = None
                path_end: tuple[float, float] | None = None
                for item in drawing.get("items", []):
                    command = item[0]
                    if command == "l":
                        first, second = _point(item[1]), _point(item[2])
                        _append_polyline(segments, [first, second], page_height, page_offset, "PDF_LINE", handle)
                        path_start = path_start or first
                        path_end = second
                    elif command == "c":
                        controls = [_point(value) for value in item[1:5]]
                        points = _cubic_points(*controls, flattening=flattening)
                        _append_polyline(segments, points, page_height, page_offset, "PDF_BEZIER", handle)
                        path_start = path_start or controls[0]
                        path_end = controls[-1]
                    elif command == "re":
                        rect = item[1]
                        points = [
                            (float(rect.x0), float(rect.y0)),
                            (float(rect.x1), float(rect.y0)),
                            (float(rect.x1), float(rect.y1)),
                            (float(rect.x0), float(rect.y1)),
                        ]
                        _append_polyline(segments, points, page_height, page_offset, "PDF_RECT", handle, close=True)
                    elif command == "qu":
                        quad = item[1]
                        points = [_point(quad.ul), _point(quad.ur), _point(quad.lr), _point(quad.ll)]
                        _append_polyline(segments, points, page_height, page_offset, "PDF_QUAD", handle, close=True)
                if drawing.get("closePath") and path_start and path_end and path_start != path_end:
                    _append_polyline(segments, [path_end, path_start], page_height, page_offset, "PDF_CLOSE", handle)

            text_data = page.get_text("dict")
            for block in text_data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        value = str(span.get("text", "")).strip()
                        if not value:
                            continue
                        origin = span.get("origin", (0.0, 0.0))
                        texts.append(
                            TextItem(
                                value=value,
                                x=float(origin[0]),
                                y=page_offset + page_height - float(origin[1]),
                                height=float(span.get("size", 0.0)),
                                handle=f"P{page_index + 1}-T{len(texts) + 1}",
                            )
                        )
            page_offset += page_height + max(page_width, page_height) * 0.05
    finally:
        document.close()

    if not segments:
        raise ValueError(
            f"PDF에서 벡터 선을 찾지 못했습니다: {path}. 스캔 이미지 PDF는 현재 입력으로 사용할 수 없습니다."
        )
    return Drawing(str(path), segments, texts)
