from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .annotate import render_annotated_pages
from .config import CompareConfig
from .dwg import ensure_dxf
from .extract import load_dxf
from .llm import describe_change
from .models import ChangeRegion, CompareResult
from .pdf_extract import load_pdf
from .render import (
    CanvasTransform,
    build_transform,
    color_overlay,
    difference_masks,
    render_segments,
    shift_mask,
)


def _save_png(path: Path, image: np.ndarray) -> None:
    # OpenCV imwrite silently fails on some Windows builds for Unicode paths.
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"PNG 인코딩에 실패했습니다: {path}")
    path.write_bytes(encoded.tobytes())


def _symmetric_residual(first: np.ndarray, second: np.ndarray, tolerance: float) -> float:
    deleted, added, _ = difference_masks(first, second, tolerance)
    numerator = int(deleted.sum() + added.sum())
    denominator = max(int(first.sum() + second.sum()), 1)
    return numerator / denominator


def _estimate_global_shift(old_mask: np.ndarray, new_mask: np.ndarray, limit_ratio: float) -> tuple[float, float]:
    old_float = cv2.GaussianBlur(old_mask.astype(np.float32), (5, 5), 0)
    new_float = cv2.GaussianBlur(new_mask.astype(np.float32), (5, 5), 0)
    try:
        proposed, response = cv2.phaseCorrelate(old_float, new_float)
    except cv2.error:
        return 0.0, 0.0
    limit = max(old_mask.shape) * limit_ratio
    candidates = [(0.0, 0.0)]
    if response > 0.02 and abs(proposed[0]) <= limit and abs(proposed[1]) <= limit:
        candidates.extend([proposed, (-proposed[0], -proposed[1])])
    return min(
        candidates,
        key=lambda shift: _symmetric_residual(shift_mask(old_mask, *shift), new_mask, 2.5),
    )


def _component_boxes(mask: np.ndarray, join_px: int) -> list[tuple[int, int, int, int, int]]:
    kernel_size = max(join_px * 2 + 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    joined = cv2.dilate(mask.astype(np.uint8), kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        boxes.append((x, y, width, height, area))
    return boxes


def _crop_bounds(
    box: tuple[int, int, int, int, int],
    shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    x, y, width, height, _ = box
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(shape[1], x + width + margin),
        min(shape[0], y + height + margin),
    )


def _suppress_minor_motion(
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    deleted: np.ndarray,
    added: np.ndarray,
    config: CompareConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    union = ((deleted > 0) | (added > 0)).astype(np.uint8)
    boxes = _component_boxes(union, config.difference_join_px)
    suppressed = np.zeros_like(union)
    count = 0
    for box in boxes:
        left, top, right, bottom = _crop_bounds(box, union.shape, config.local_search_margin_px)
        old_crop = old_mask[top:bottom, left:right]
        new_crop = new_mask[top:bottom, left:right]
        if min(old_crop.shape) < 4 or old_crop.sum() < 5 or new_crop.sum() < 5:
            continue
        before = _symmetric_residual(old_crop, new_crop, config.match_tolerance_px)
        old_float = cv2.GaussianBlur(old_crop.astype(np.float32), (3, 3), 0)
        new_float = cv2.GaussianBlur(new_crop.astype(np.float32), (3, 3), 0)
        candidate_shifts: list[tuple[float, float, float]] = []
        try:
            shift, response = cv2.phaseCorrelate(old_float, new_float)
            candidate_shifts.append((float(shift[0]), float(shift[1]), float(response)))
        except cv2.error:
            pass

        old_points = np.argwhere(old_crop > 0)
        new_points = np.argwhere(new_crop > 0)
        if len(old_points) and len(new_points):
            old_center = old_points.mean(axis=0)
            new_center = new_points.mean(axis=0)
            candidate_shifts.append(
                (
                    float(new_center[1] - old_center[1]),
                    float(new_center[0] - old_center[0]),
                    1.0,
                )
            )
        deleted_points = np.argwhere(deleted[top:bottom, left:right] > 0)
        added_points = np.argwhere(added[top:bottom, left:right] > 0)
        if len(deleted_points) and len(added_points):
            deleted_center = deleted_points.mean(axis=0)
            added_center = added_points.mean(axis=0)
            candidate_shifts.append(
                (
                    float(added_center[1] - deleted_center[1]),
                    float(added_center[0] - deleted_center[0]),
                    1.0,
                )
            )
        viable = [
            candidate
            for candidate in candidate_shifts
            if candidate[2] >= 0.04
            and (candidate[0] ** 2 + candidate[1] ** 2) ** 0.5 <= config.minor_motion_limit_px
        ]
        if not viable:
            continue
        after, best_shift = min(
            (
                _symmetric_residual(
                    shift_mask(old_crop, candidate[0], candidate[1]),
                    new_crop,
                    config.match_tolerance_px,
                ),
                candidate,
            )
            for candidate in viable
        )
        if after <= config.minor_residual_ratio and after < before * 0.55:
            x, y, width, height, _ = box
            suppressed[y : y + height, x : x + width] = 1
            count += 1
    return deleted * (1 - suppressed), added * (1 - suppressed), count


def _drawing_bbox(
    transform: CanvasTransform, left: int, top: int, right: int, bottom: int
) -> tuple[float, float, float, float]:
    x1, y1 = transform.pixel_to_point(left, bottom)
    x2, y2 = transform.pixel_to_point(right, top)
    return round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)


def _mask_components(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    joined = cv2.dilate(mask.astype(np.uint8), kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    return sum(1 for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= 4)


def _closed_contours(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    contours, hierarchy = cv2.findContours(
        (mask.astype(np.uint8) * 255), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    return sum(
        1
        for index, contour in enumerate(contours)
        if hierarchy[0][index][3] >= 0 and abs(cv2.contourArea(contour)) >= 3.0
    )


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    points = np.argwhere(mask > 0)
    if not len(points):
        return None
    center = points.mean(axis=0)
    return float(center[1]), float(center[0])


def _region_evidence(
    old_crop: np.ndarray,
    new_crop: np.ndarray,
    deleted_crop: np.ndarray,
    added_crop: np.ndarray,
    transform: CanvasTransform,
    width_px: int,
    height_px: int,
    tolerance_px: float,
) -> dict[str, object]:
    deleted_count = int(deleted_crop.sum())
    added_count = int(added_crop.sum())
    total = max(deleted_count + added_count, 1)
    evidence: dict[str, object] = {
        "width": round(width_px / transform.scale, 3),
        "height": round(height_px / transform.scale, 3),
        "added_length": round(added_count / transform.scale, 3),
        "deleted_length": round(deleted_count / transform.scale, 3),
        "added_share": round(added_count / total, 4),
        "deleted_share": round(deleted_count / total, 4),
        "added_components": _mask_components(added_crop),
        "deleted_components": _mask_components(deleted_crop),
        "added_closed_contours": _closed_contours(added_crop),
        "deleted_closed_contours": _closed_contours(deleted_crop),
    }
    old_center = _centroid(deleted_crop)
    new_center = _centroid(added_crop)
    if old_center and new_center and deleted_count >= 8 and added_count >= 8:
        dx_px = new_center[0] - old_center[0]
        dy_px = new_center[1] - old_center[1]
        shifted = shift_mask(old_crop, dx_px, dy_px)
        residual = _symmetric_residual(shifted, new_crop, tolerance_px)
        evidence["estimated_local_shift"] = {
            "dx": round(dx_px / transform.scale, 3),
            "dy": round(-dy_px / transform.scale, 3),
        }
        evidence["aligned_residual_ratio"] = round(residual, 4)
    return evidence


def _analysis_panel(old_crop: np.ndarray, new_crop: np.ndarray, overlay_crop: np.ndarray) -> np.ndarray:
    old_view = np.repeat((old_crop * 190)[:, :, None], 3, axis=2).astype(np.uint8)
    new_view = np.repeat((new_crop * 190)[:, :, None], 3, axis=2).astype(np.uint8)
    panels = []
    for label, panel in (("OLD", old_view), ("NEW", new_view), ("OVERLAY", overlay_crop)):
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (max(panel.shape[1] - 1, 0), min(22, panel.shape[0] - 1)), (20, 28, 34), -1)
        cv2.putText(panel, label, (5, min(16, panel.shape[0] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def _extract_regions(
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    deleted: np.ndarray,
    added: np.ndarray,
    overlay: np.ndarray,
    transform: CanvasTransform,
    output_dir: Path,
    config: CompareConfig,
    use_llm: bool = False,
) -> list[ChangeRegion]:
    union = ((deleted > 0) | (added > 0)).astype(np.uint8)
    boxes = _component_boxes(union, config.difference_join_px)
    regions: list[ChangeRegion] = []
    crops_dir = output_dir / "regions"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for box in boxes:
        left, top, right, bottom = _crop_bounds(box, union.shape, 5)
        raw_union = union[top:bottom, left:right]
        changed_pixels = int(raw_union.sum())
        if changed_pixels < config.min_region_pixels:
            continue
        if min(right - left, bottom - top) < config.min_region_dimension_px:
            continue
        old_count = int(deleted[top:bottom, left:right].sum())
        new_count = int(added[top:bottom, left:right].sum())
        area = max((right - left) * (bottom - top), 1)
        ratio = changed_pixels / area
        region_id = f"CHG-{len(regions) + 1:04d}"
        crop_path = crops_dir / f"{region_id}.png"
        _save_png(crop_path, overlay[top:bottom, left:right])
        old_crop = old_mask[top:bottom, left:right]
        new_crop = new_mask[top:bottom, left:right]
        deleted_crop = deleted[top:bottom, left:right]
        added_crop = added[top:bottom, left:right]
        evidence = _region_evidence(
            old_crop,
            new_crop,
            deleted_crop,
            added_crop,
            transform,
            right - left,
            bottom - top,
            config.match_tolerance_px,
        )
        stats = {
            "added_pixels": new_count,
            "deleted_pixels": old_count,
            "changed_pixels": changed_pixels,
            "area_pixels": area,
            "change_ratio": round(ratio, 6),
            **evidence,
        }
        analysis_path = crops_dir / f"{region_id}_analysis.png"
        _save_png(
            analysis_path,
            _analysis_panel(old_crop, new_crop, overlay[top:bottom, left:right]),
        )
        classification, description, confidence = describe_change(analysis_path, stats, use_llm=use_llm)
        regions.append(
            ChangeRegion(
                id=region_id,
                bbox_pixels=(left, top, right, bottom),
                bbox_drawing=_drawing_bbox(transform, left, top, right, bottom),
                deleted_pixels=old_count,
                added_pixels=new_count,
                changed_pixels=changed_pixels,
                area_pixels=area,
                change_ratio=round(ratio, 6),
                classification=classification,
                description=description,
                confidence=round(confidence, 3),
                evidence=evidence,
                image=(Path("regions") / crop_path.name).as_posix(),
            )
        )
    regions.sort(key=lambda region: region.changed_pixels, reverse=True)
    return regions


def compare_files(
    old_file: Path,
    new_file: Path,
    output_dir: Path,
    config: CompareConfig | None = None,
    *,
    use_llm: bool = False,
) -> CompareResult:
    config = config or CompareConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    conversion_dir = output_dir / "converted"
    if old_file.suffix.lower() == ".pdf":
        old = load_pdf(old_file, max(config.curve_flattening, 0.5))
    else:
        old_dxf = ensure_dxf(old_file, conversion_dir / "old")
        old = load_dxf(old_dxf, config.curve_flattening)
    if new_file.suffix.lower() == ".pdf":
        new = load_pdf(new_file, max(config.curve_flattening, 0.5))
    else:
        new_dxf = ensure_dxf(new_file, conversion_dir / "new")
        new = load_dxf(new_dxf, config.curve_flattening)
    transform = build_transform([old, new], config.canvas_size, config.padding)
    old_mask = render_segments(old.segments, transform, config.line_thickness)
    new_mask = render_segments(new.segments, transform, config.line_thickness)

    shift_x, shift_y = _estimate_global_shift(old_mask, new_mask, config.global_shift_limit_ratio)
    aligned_old = shift_mask(old_mask, shift_x, shift_y)
    deleted, added, common = difference_masks(aligned_old, new_mask, config.match_tolerance_px)
    deleted, added, suppressed = _suppress_minor_motion(
        aligned_old, new_mask, deleted, added, config
    )
    common = ((aligned_old > 0) | (new_mask > 0)).astype(np.uint8) * (1 - ((deleted > 0) | (added > 0)))
    overlay = color_overlay(deleted, added, common)
    overlay_path = output_dir / "overlay.png"
    _save_png(overlay_path, overlay)
    regions = _extract_regions(
        aligned_old, new_mask, deleted, added, overlay, transform, output_dir, config, use_llm
    )

    shift_drawing = (shift_x / transform.scale, -shift_y / transform.scale)
    result = CompareResult(
        old_file=str(old_file),
        new_file=str(new_file),
        output_dir=output_dir,
        global_shift_drawing=(round(shift_drawing[0], 6), round(shift_drawing[1], 6)),
        suppressed_minor_regions=suppressed,
        regions=regions,
        overlay_path=overlay_path,
        annotated_pages=render_annotated_pages(new_file, regions, output_dir),
    )
    from .report import write_report

    result.report_path = write_report(result)
    (output_dir / "result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
