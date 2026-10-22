"""Trace the green and red dashed vector centerlines of a drawing PDF.

Short gaps matching the source dash pattern become continuous CAD lines.
Long aligned gaps are exported on a separate inferred layer for review.
The source PDF is unchanged.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import ezdxf
import pymupdf


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_inputs" / "DWG 객체 변환 샘플.pdf"
OUTPUT = ROOT / "green_centerline_trace"
SHORT_GAP_PT = 3.5
MAX_INFERRED_GAP_PT = 250.0
MIN_AXIS_SEGMENTS = 5
MIN_AXIS_SPAN_PT = 60.0
PDF_TO_DWG_SCALE = 1.0 / 72.0


def green(color: object) -> bool:
    return (isinstance(color, (tuple, list)) and len(color) == 3
            and color[1] > 0.90 and color[2] < 0.10
            and color[0] < color[1] - 0.15)


def red_centerline(color: object, width: float, item_count: int) -> bool:
    return (isinstance(color, (tuple, list)) and len(color) == 3
            and color[0] > 0.95 and color[1] < 0.05 and color[2] < 0.05
            and width > 1.5 and item_count >= MIN_AXIS_SEGMENTS)


def merged_intervals(intervals: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if result and start - result[-1][1] <= gap:
            result[-1] = result[-1][0], max(result[-1][1], end)
        else:
            result.append((start, end))
    return result


def extract(page: pymupdf.Page) -> tuple[list[dict], list[dict], dict]:
    axes: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)
    green_paths = 0
    red_paths = 0
    source_dashes = 0
    for drawing in page.get_drawings():
        is_green = green(drawing.get("color"))
        is_red = red_centerline(drawing.get("color"), float(drawing.get("width") or 0),
                                len(drawing.get("items", [])))
        if not (is_green or is_red):
            continue
        green_paths += int(is_green)
        red_paths += int(is_red)
        for item in drawing.get("items", []):
            if item[0] != "l":
                continue
            first, second = item[1], item[2]
            if abs(first.y - second.y) <= 0.15:
                key = "H", round((first.y + second.y) / 2, 1)
                axes[key].append(tuple(sorted((float(first.x), float(second.x)))))
                source_dashes += 1
            elif abs(first.x - second.x) <= 0.15:
                key = "V", round((first.x + second.x) / 2, 1)
                axes[key].append(tuple(sorted((float(first.y), float(second.y)))))
                source_dashes += 1

    confirmed: list[dict] = []
    inferred: list[dict] = []
    for (orientation, position), source_intervals in sorted(axes.items()):
        start = min(a for a, _ in source_intervals)
        end = max(b for _, b in source_intervals)
        if len(source_intervals) < MIN_AXIS_SEGMENTS or end - start < MIN_AXIS_SPAN_PT:
            continue  # Tiny nearby marks are not a long green route.
        runs = merged_intervals(source_intervals, SHORT_GAP_PT)
        for a, b in runs:
            confirmed.append({"axis": orientation, "position_pdf_pt": position,
                              "start_pdf_pt": a, "end_pdf_pt": b,
                              "source_dash_count": sum(x <= b and y >= a for x, y in source_intervals)})
        for left, right in zip(runs, runs[1:]):
            distance = right[0] - left[1]
            if SHORT_GAP_PT < distance <= MAX_INFERRED_GAP_PT:
                inferred.append({"axis": orientation, "position_pdf_pt": position,
                                 "start_pdf_pt": left[1], "end_pdf_pt": right[0],
                                 "gap_pdf_pt": round(distance, 3)})
    return confirmed, inferred, {"green_pdf_paths": green_paths,
                                  "red_centerline_pdf_paths": red_paths,
                                  "source_dash_segments": source_dashes,
                                  "route_axes": len({(r["axis"], r["position_pdf_pt"]) for r in confirmed})}


def pdf_points(row: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    position = row["position_pdf_pt"]
    a, b = row["start_pdf_pt"], row["end_pdf_pt"]
    return ((a, position), (b, position)) if row["axis"] == "H" else ((position, a), (position, b))


def export_dxf(confirmed: list[dict], inferred: list[dict], page_height: float) -> Path:
    drawing = ezdxf.new("R2018")
    drawing.layers.new("GREEN_CENTERLINE_CONFIRMED", dxfattribs={"color": 4})
    drawing.layers.new("GREEN_CENTERLINE_INFERRED", dxfattribs={"color": 30})
    model = drawing.modelspace()
    def to_cad(point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * PDF_TO_DWG_SCALE, (page_height - point[1]) * PDF_TO_DWG_SCALE
    for row in confirmed:
        model.add_lwpolyline([to_cad(point) for point in pdf_points(row)],
                             dxfattribs={"layer": "GREEN_CENTERLINE_CONFIRMED"})
    for row in inferred:
        model.add_lwpolyline([to_cad(point) for point in pdf_points(row)],
                             dxfattribs={"layer": "GREEN_CENTERLINE_INFERRED"})
    result = OUTPUT / "green_centerlines.dxf"
    drawing.saveas(result)
    return result


def export_overlay(source: Path, confirmed: list[dict], inferred: list[dict]) -> Path:
    with pymupdf.open(source) as doc:
        page = doc[0]
        for row in confirmed:
            page.draw_line(*pdf_points(row), color=(1, 0.0, 0.5), width=3.5)
        for row in inferred:
            page.draw_line(*pdf_points(row), color=(1, 0.55, 0), width=3.5,
                           dashes="[8 5] 0")
        preview = OUTPUT / "green_centerlines_overlay.pdf"
        doc.save(preview, garbage=4, deflate=True)
        overview = page.get_pixmap(matrix=pymupdf.Matrix(0.35, 0.35), alpha=False)
        overview.save(OUTPUT / "overview.png")
        clip = pymupdf.Rect(430, 1690, 1500, 1795)
        detail = page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), clip=clip, alpha=False)
        detail.save(OUTPUT / "main_line_detail.png")
        vertical_clip = pymupdf.Rect(3290, 1080, 3420, 1810)
        vertical = page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4),
                                  clip=vertical_clip, alpha=False)
        vertical.save(OUTPUT / "vertical_line_detail.png")
        return preview


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(SOURCE) as doc:
        if len(doc) != 1:
            raise ValueError("This sample trace expects one PDF page")
        confirmed, inferred, counts = extract(doc[0])
        page_height = doc[0].rect.height
    if not confirmed:
        raise ValueError("No green centerline candidates found")
    dxf = export_dxf(confirmed, inferred, page_height)
    overlay = export_overlay(SOURCE, confirmed, inferred)
    report = {"source_pdf": str(SOURCE), **counts,
              "confirmed_runs": len(confirmed), "inferred_long_gaps": len(inferred),
              "short_gap_threshold_pdf_pt": SHORT_GAP_PT,
              "inferred_gap_max_pdf_pt": MAX_INFERRED_GAP_PT,
              "confirmed": confirmed, "inferred": inferred,
              "dxf": str(dxf), "overlay_pdf": str(overlay)}
    (OUTPUT / "trace_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("green_pdf_paths", "red_centerline_pdf_paths",
                       "source_dash_segments", "route_axes",
                       "confirmed_runs", "inferred_long_gaps")}, indent=2))


if __name__ == "__main__":
    main()
