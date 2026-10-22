import unittest
import cv2
import numpy as np

from table_confusion_guard import open_c_shape, supported_c


class ConfusionGuardTests(unittest.TestCase):
    def test_open_shape_and_closed_loop_are_distinct(self):
        image = np.full((64, 64), 255, np.uint8)
        points = np.array([[50, 16], [42, 8], [18, 8], [10, 16], [10, 48], [18, 56], [42, 56], [50, 48]])
        cv2.polylines(image, [points], False, 0, 2)
        self.assertTrue(open_c_shape(image)['compatible'])
        cv2.line(image, (50, 16), (50, 48), 0, 2)
        self.assertFalse(open_c_shape(image)['compatible'])

    def test_models_geometry_and_numeric_protection_are_all_required(self):
        glyph = dict(text='C', confidence=.98)
        views = [dict(text='C', score=.999)]*2
        self.assertTrue(supported_c('0', glyph, views, dict(compatible=True)))
        self.assertFalse(supported_c('0', glyph, views, dict(compatible=True), protected=True))
        self.assertFalse(supported_c('0', glyph, views, dict(compatible=False)))
        self.assertFalse(supported_c('0', glyph, views[:1], dict(compatible=True)))
        self.assertFalse(supported_c('0', dict(text='0', confidence=.99), views, dict(compatible=True)))
        self.assertFalse(supported_c('0', glyph, [dict(text='O', score=.999)]*2, dict(compatible=True)))


if __name__ == '__main__':
    unittest.main()
