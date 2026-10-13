from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import Drawing, Segment


@dataclass(frozen=True)
class CanvasTransform:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    scale: float
    width: int
    height: int
    padding: int

    def point_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        px = int(round((x - self.min_x) * self.scale)) + self.padding
        py = self.height - self.padding - int(round((y - self.min_y) * self.scale))
        return px, py

    def pixel_to_point(self, px: float, py: float) -> tuple[float, float]:
        x = (px - self.padding) / self.scale + self.min_x
        y = (self.height - self.padding - py) / self.scale + self.min_y
        return x, y


def build_transform(drawings: list[Drawing], canvas_size: int, padding: int) -> CanvasTransform:
    bounds = [drawing.bounds for drawing in drawings]
    min_x = min(b[0] for b in bounds)
    min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds)
    max_y = max(b[3] for b in bounds)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    usable = max(canvas_size - 2 * padding, 100)
    scale = usable / max(span_x, span_y)
    width = int(round(span_x * scale)) + 2 * padding
    height = int(round(span_y * scale)) + 2 * padding
    return CanvasTransform(min_x, min_y, max_x, max_y, scale, width, height, padding)


def render_segments(
    segments: list[Segment],
    transform: CanvasTransform,
    thickness: int = 1,
) -> np.ndarray:
    canvas = np.zeros((transform.height, transform.width), dtype=np.uint8)
    for segment in segments:
        first = transform.point_to_pixel(segment.x1, segment.y1)
        second = transform.point_to_pixel(segment.x2, segment.y2)
        cv2.line(canvas, first, second, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return (canvas > 32).astype(np.uint8)


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def distance_to_ink(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)


def difference_masks(
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    tolerance_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_distance_to_new = distance_to_ink(new_mask)
    new_distance_to_old = distance_to_ink(old_mask)
    deleted = ((old_mask > 0) & (old_distance_to_new > tolerance_px)).astype(np.uint8)
    added = ((new_mask > 0) & (new_distance_to_old > tolerance_px)).astype(np.uint8)
    common = ((old_mask > 0) & (old_distance_to_new <= tolerance_px)).astype(np.uint8)
    common |= ((new_mask > 0) & (new_distance_to_old <= tolerance_px)).astype(np.uint8)
    return deleted, added, common


def color_overlay(deleted: np.ndarray, added: np.ndarray, common: np.ndarray) -> np.ndarray:
    image = np.full((*deleted.shape, 3), 25, dtype=np.uint8)
    image[common > 0] = (155, 155, 155)
    image[deleted > 0] = (30, 30, 235)  # BGR red
    image[added > 0] = (40, 205, 40)  # BGR green
    both = (deleted > 0) & (added > 0)
    image[both] = (30, 200, 230)
    return image
