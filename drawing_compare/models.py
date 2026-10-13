from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "LINE"
    handle: str = ""

    @property
    def length(self) -> float:
        return ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5


@dataclass(frozen=True)
class TextItem:
    value: str
    x: float
    y: float
    height: float
    handle: str = ""


@dataclass
class Drawing:
    source: str
    segments: list[Segment]
    texts: list[TextItem] = field(default_factory=list)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if not self.segments:
            return (0.0, 0.0, 1.0, 1.0)
        xs = [p for s in self.segments for p in (s.x1, s.x2)]
        ys = [p for s in self.segments for p in (s.y1, s.y2)]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class ChangeRegion:
    id: str
    bbox_pixels: tuple[int, int, int, int]
    bbox_drawing: tuple[float, float, float, float]
    deleted_pixels: int
    added_pixels: int
    changed_pixels: int
    area_pixels: int
    change_ratio: float
    classification: str
    description: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompareResult:
    old_file: str
    new_file: str
    output_dir: Path
    global_shift_drawing: tuple[float, float]
    suppressed_minor_regions: int
    regions: list[ChangeRegion]
    overlay_path: Path
    annotated_pages: list[str] = field(default_factory=list)
    report_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_file": self.old_file,
            "new_file": self.new_file,
            "global_shift": {
                "dx": self.global_shift_drawing[0],
                "dy": self.global_shift_drawing[1],
            },
            "suppressed_minor_regions": self.suppressed_minor_regions,
            "regions": [region.to_dict() for region in self.regions],
            "overlay": self.overlay_path.name,
            "annotated_pages": self.annotated_pages,
            "report": self.report_path.name if self.report_path else None,
        }
