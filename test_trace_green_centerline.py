import unittest
from types import SimpleNamespace

from trace_green_centerline import green, extract


class CenterlineColorTests(unittest.TestCase):
    def test_green_and_yellow_green_are_allowed_but_yellow_and_other_inks_are_not(self):
        self.assertTrue(green((0, 1, 0)))
        self.assertTrue(green((.749, 1, 0)))
        for color in ((1, 1, 0), (1, 0, 0), (.251, .373, .502), (0, 0, 0), None):
            self.assertFalse(green(color))

    def test_color_change_inside_same_axis_does_not_create_false_gap(self):
        def drawing(color, start, count):
            return dict(color=color, width=2, items=[('l', SimpleNamespace(x=10., y=start+i*10),
                                   SimpleNamespace(x=10., y=start+i*10+8)) for i in range(count)])
        page = SimpleNamespace(get_drawings=lambda:[drawing((0, 1, 0), 0, 8),
                                                    drawing((.749, 1, 0), 80, 8),
                                                    drawing((0, 1, 0), 160, 8)])
        confirmed, inferred, _ = extract(page)
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(inferred, [])
        self.assertEqual(confirmed[0]['source_dash_count'], 24)


if __name__ == '__main__':
    unittest.main()
