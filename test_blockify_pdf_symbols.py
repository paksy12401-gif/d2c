import math
import unittest
from unittest.mock import patch

import blockify_pdf_symbols as blockify


class ArcBoundsTests(unittest.TestCase):
    def test_radian_arc_crosses_quadrant_extrema(self):
        bounds = blockify.entity_bounds({
            "type": "ARC", "cx": "1", "cy": "2", "r": "2",
            "start": str(math.pi / 2), "end": str(3 * math.pi / 2),
        })
        for actual, expected in zip(bounds, (-1, 0, 1, 4)):
            self.assertAlmostEqual(actual, expected)

    def test_radian_arc_wraps_across_zero(self):
        bounds = blockify.entity_bounds({
            "type": "ARC", "cx": "1", "cy": "2", "r": "2",
            "start": str(3 * math.pi / 2), "end": str(math.pi / 2),
        })
        for actual, expected in zip(bounds, (1, 0, 3, 4)):
            self.assertAlmostEqual(actual, expected)

    def test_review_candidate_contains_complete_arc(self):
        row = {
            "type": "ARC", "handle": "A1", "layer": "PDF_Geometry", "color": "7",
            "cx": "1", "cy": "2", "r": "2",
            "start": str(math.pi / 2), "end": str(3 * math.pi / 2),
        }
        row["bounds"] = blockify.entity_bounds(row)
        detection = {"class_name": "part", "status": "review",
                     "bbox_pdf_pt": [-72, 432, 72.01, 720]}
        with patch.object(blockify, "INCLUDE_REVIEW", True):
            groups = blockify.assign_entities([detection], [row], 720)
        self.assertEqual(groups[0]["handles"], ["A1"])
        with patch.object(blockify, "INCLUDE_REVIEW", False):
            self.assertEqual(blockify.assign_entities([detection], [row], 720), [])

    def test_wide_symbol_flange_is_not_mistaken_for_pipe(self):
        rows = [
            {"handle": "flange", "type": "LWPOLYLINE", "layer": "PDF_Geometry",
             "color": "256", "bounds": (0.05, 0.45, 0.95, 0.46)},
            {"handle": "mount", "type": "LWPOLYLINE", "layer": "PDF_Geometry",
             "color": "256", "bounds": (-0.15, 0.2, 0.10, 0.7)},
        ]
        detection = {"class_name": "7", "status": "strong_candidate",
                     "bbox_pdf_pt": [0, 648, 72, 720]}
        groups = blockify.assign_entities([detection], rows, 720)
        self.assertEqual(groups[0]["handles"], ["flange", "mount"])

    def test_pipe_outline_is_left_outside_symbol_block(self):
        rows = [
            {"handle": "pipe", "type": "LWPOLYLINE", "layer": "PDF_Geometry",
             "color": "256", "bounds": (0.2, 0.2, 0.6, 0.6)},
            {"handle": "fastener", "type": "LWPOLYLINE", "layer": "PDF_Geometry",
             "color": "256", "bounds": (0.6, 0.2, 0.8, 0.6)},
        ]
        detection = {"class_name": "7", "status": "strong_candidate",
                     "bbox_pdf_pt": [0, 648, 72, 720]}
        with patch.dict(blockify.PIPE_COMPONENT_HANDLES_BY_BLOCK,
                        {"SYM_7_001": ("pipe",)}):
            groups = blockify.assign_entities([detection], rows, 720)
        self.assertEqual(groups[0]["handles"], ["fastener"])
        self.assertEqual(groups[0]["left_as_pipe"], ["pipe"])

    def test_pipe_endcap_is_not_split_from_pipe_outline(self):
        rows = [
            {"handle": handle, "type": "LWPOLYLINE", "layer": "PDF_Geometry",
             "color": "256", "bounds": bounds}
            for handle, bounds in [
                ("pipe", (0.1, 0.2, 0.9, 0.5)),
                ("endcap", (0.85, 0.2, 0.95, 0.5)),
                ("fastener", (0.4, 0.4, 0.6, 0.8)),
            ]
        ]
        detection = {"class_name": "11", "status": "strong_candidate",
                     "bbox_pdf_pt": [0, 648, 72, 720]}
        with patch.dict(blockify.PIPE_COMPONENT_HANDLES_BY_BLOCK,
                        {"SYM_11_001": ("pipe", "endcap")}):
            groups = blockify.assign_entities([detection], rows, 720)
        self.assertEqual(groups[0]["handles"], ["fastener"])
        self.assertEqual(groups[0]["left_as_pipe"], ["endcap", "pipe"])

    def test_similar_symbols_keep_their_own_vector_definitions(self):
        groups = [
            {"block_name": "SYM_11_037", "class_name": "11", "handles": ["a", "b"]},
            {"block_name": "SYM_11_045", "class_name": "11", "handles": ["c", "d", "e"]},
        ]
        blockify.plan_definition_reuse(groups, None)
        self.assertEqual([g["definition_name"] for g in groups],
                         ["SYM_11_037", "SYM_11_045"])


if __name__ == "__main__":
    unittest.main()
