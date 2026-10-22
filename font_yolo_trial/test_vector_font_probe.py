import unittest

from vector_font_probe import group_glyphs, parse_linear_path, load_candidates
from types import SimpleNamespace


IDENTITY = [1, 0, 0, 1, 0, 0]


class VectorSegmentationChecks(unittest.TestCase):
    def test_moveto_keeps_separate_runs_at_touching_endpoints(self):
        runs = parse_linear_path('M0 0L5 0 M5 0L10 0', IDENTITY)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(group_glyphs(runs)), 2)

    def test_pen_lift_inside_one_letter_can_be_grouped(self):
        runs = parse_linear_path('M0 10V0H5 M0 5H4', IDENTITY)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(group_glyphs(runs)), 1)

    def test_relative_commands_implicit_lines_and_transform(self):
        runs = parse_linear_path('m1 2 3 0 v2 h-3 z', [2, 0, 0, -2, 10, 20])
        self.assertEqual(runs[0][0], (12, 16, 18, 16))
        self.assertEqual(runs[0][-1], (12, 12, 12, 16))

    def test_unsupported_curve_is_skipped_without_partial_geometry(self):
        self.assertEqual(parse_linear_path('M0 0L1 1C2 2 3 3 4 4', IDENTITY), [])

    def test_large_candidates_can_be_requested_without_changing_legacy_default(self):
        page = SimpleNamespace(get_svg_image=lambda **kwargs:
            '<svg><path stroke="black" d="M0 0V100H30 M100 0V100H130 M200 0V100H230"/></svg>')
        self.assertEqual(load_candidates(page), [])
        self.assertEqual(len(load_candidates(page, min_cap=1, max_cap=150)), 1)


if __name__ == '__main__':
    unittest.main()
