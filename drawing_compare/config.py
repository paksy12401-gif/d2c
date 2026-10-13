from dataclasses import dataclass


@dataclass
class CompareConfig:
    canvas_size: int = 2400
    padding: int = 30
    line_thickness: int = 1
    curve_flattening: float = 0.1
    match_tolerance_px: float = 2.5
    global_shift_limit_ratio: float = 0.08
    difference_join_px: int = 14
    local_search_margin_px: int = 28
    minor_motion_limit_px: float = 24.0
    minor_residual_ratio: float = 0.10
    min_region_pixels: int = 12
    min_region_dimension_px: int = 3
    major_change_ratio: float = 0.08
    llm_min_changed_pixels: int = 60
