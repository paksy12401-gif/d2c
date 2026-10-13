from __future__ import annotations

import math
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

from .models import ChangeRegion


CHANGE_COLORS: dict[str, tuple[int, int, int]] = {
    "ADDED": (24, 156, 86),
    "DELETED": (220, 42, 48),
    "MOVED": (37, 99, 235),
    "MOVED_AND_MODIFIED": (237, 126, 28),
    "GEOMETRY_MODIFIED": (139, 69, 224),
    "TEXT_CHANGED": (184, 54, 184),
    "DIMENSION_CHANGED": (184, 54, 184),
    "CONNECTION_CHANGED": (139, 69, 224),
}


def _change_color(classification: str) -> tuple[int, int, int]:
    return CHANGE_COLORS.get(classification, (139, 69, 224))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path(r"C:\Windows\Fonts\malgun.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _short_label(region: ChangeRegion) -> str:
    evidence = region.evidence
    added_share = float(evidence.get("added_share", 0.0))
    deleted_share = float(evidence.get("deleted_share", 0.0))
    labels = {
        "ADDED": "신규 형상 추가",
        "DELETED": "기존 형상 삭제",
        "MOVED": "위치 이동",
        "MOVED_AND_MODIFIED": "이동 후 형상 변경",
        "TEXT_CHANGED": "문자 표기 변경",
        "DIMENSION_CHANGED": "치수 변경",
        "CONNECTION_CHANGED": "연결 구조 변경",
    }
    label = labels.get(region.classification)
    if not label:
        if added_share >= 0.85:
            label = "형상 대규모 추가"
        elif deleted_share >= 0.85:
            label = "형상 대규모 삭제"
        else:
            label = "형상 구성 변경"
    if region.classification in {"GEOMETRY_MODIFIED", "MOVED_AND_MODIFIED"}:
        label += f" (추가 {added_share:.0%}, 삭제 {deleted_share:.0%})"
    return f"{region.id}  {label}"


def _page_offsets(document: pymupdf.Document) -> list[tuple[float, float, float]]:
    offsets: list[tuple[float, float, float]] = []
    offset = 0.0
    for page in document:
        height = float(page.rect.height)
        width = float(page.rect.width)
        offsets.append((offset, width, height))
        offset += height + max(width, height) * 0.05
    return offsets


def _region_for_page(
    region: ChangeRegion,
    offset: float,
    page_height: float,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = region.bbox_drawing
    center_y = (y1 + y2) / 2
    if center_y < offset or center_y > offset + page_height:
        return None
    top = page_height - (y2 - offset)
    bottom = page_height - (y1 - offset)
    return x1, top, x2, bottom


def render_annotated_pages(
    pdf_path: Path,
    regions: list[ChangeRegion],
    output_dir: Path,
    max_regions: int = 18,
    zoom: float = 2.0,
) -> list[str]:
    if pdf_path.suffix.lower() != ".pdf":
        return []
    selected = [region for region in regions if region.changed_pixels >= 50][:max_regions]
    if not selected:
        return []
    document = pymupdf.open(pdf_path)
    output_paths: list[str] = []
    # Large CAD sheets can be over 12,000 pt wide; cap preview memory.
    zoom = min(zoom, min(math.sqrt(16_000_000 / (page.rect.width * page.rect.height)) for page in document))
    font = _font(max(18, int(12 * zoom)))
    try:
        offsets = _page_offsets(document)
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            draw = ImageDraw.Draw(image)
            offset, _, page_height = offsets[page_index]
            occupied: list[tuple[int, int, int, int]] = []
            page_regions: list[tuple[ChangeRegion, tuple[float, float, float, float]]] = []
            for region in selected:
                page_box = _region_for_page(region, offset, page_height)
                if page_box:
                    page_regions.append((region, page_box))
            for region, page_box in page_regions:
                color = _change_color(region.classification)
                left, top, right, bottom = [int(round(value * zoom)) for value in page_box]
                left = max(0, min(image.width - 1, left))
                right = max(left + 1, min(image.width - 1, right))
                top = max(0, min(image.height - 1, top))
                bottom = max(top + 1, min(image.height - 1, bottom))
                draw.rectangle((left, top, right, bottom), outline=color, width=max(2, int(1.5 * zoom)))

                label = _short_label(region)
                text_box = draw.textbbox((0, 0), label, font=font)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]
                label_left = min(max(left, 0), max(image.width - text_width - 12, 0))
                label_top = top - text_height - 12
                if label_top < 0:
                    label_top = min(bottom + 5, image.height - text_height - 10)
                candidate = (label_left, label_top, label_left + text_width + 10, label_top + text_height + 8)
                attempts = 0
                while any(
                    not (candidate[2] < box[0] or candidate[0] > box[2] or candidate[3] < box[1] or candidate[1] > box[3])
                    for box in occupied
                ) and attempts < 8:
                    label_top = min(label_top + text_height + 12, image.height - text_height - 10)
                    candidate = (label_left, label_top, label_left + text_width + 10, label_top + text_height + 8)
                    attempts += 1
                occupied.append(candidate)
                draw.rectangle(candidate, fill=(255, 255, 255), outline=color, width=2)
                draw.text((label_left + 5, label_top + 3), label, font=font, fill=color)

            if page_regions:
                legend_items = [
                    ("추가", CHANGE_COLORS["ADDED"]),
                    ("삭제", CHANGE_COLORS["DELETED"]),
                    ("이동", CHANGE_COLORS["MOVED"]),
                    ("이동+수정", CHANGE_COLORS["MOVED_AND_MODIFIED"]),
                    ("형상/문자/치수 변경", CHANGE_COLORS["GEOMETRY_MODIFIED"]),
                ]
                legend_font = _font(max(16, int(10 * zoom)))
                row_height = max(25, int(17 * zoom))
                legend_width = max(230, int(170 * zoom))
                legend_height = row_height * len(legend_items) + 16
                legend_left = max(8, image.width - legend_width - 12)
                legend_top = 10
                draw.rectangle(
                    (legend_left, legend_top, legend_left + legend_width, legend_top + legend_height),
                    fill=(255, 255, 255),
                    outline=(90, 90, 90),
                    width=1,
                )
                for index, (name, color) in enumerate(legend_items):
                    y = legend_top + 8 + index * row_height
                    draw.rectangle((legend_left + 9, y + 3, legend_left + 27, y + 21), fill=color)
                    draw.text((legend_left + 36, y), name, font=legend_font, fill=(35, 35, 35))

            if page_regions:
                filename = f"annotated_page_{page_index + 1}.png"
                image.save(output_dir / filename, optimize=True)
                output_paths.append(filename)
    finally:
        document.close()
    return output_paths
