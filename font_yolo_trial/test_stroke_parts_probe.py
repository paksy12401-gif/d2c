import unittest

import cv2
import numpy as np

from stroke_parts_probe import verify_h


def draw(segments,thickness=2):
    image=np.full((120,120,3),255,np.uint8)
    for x0,y0,x1,y1 in segments:
        cv2.line(image,(x0,y0),(x1,y1),(0,0,0),thickness)
    return image


class StrokePartsTests(unittest.TestCase):
    def test_connected_h_at_multiple_thicknesses(self):
        for width in (1,2,4):
            with self.subTest(width=width):
                result=verify_h(draw([(20,10,20,100),(90,10,90,100),(20,55,90,55)],width))
                self.assertTrue(result['compatible'])
                self.assertGreaterEqual(result['explained_ink'],.9)

    def test_two_stems_are_insufficient(self):
        self.assertFalse(verify_h(draw([(20,10,20,100),(90,10,90,100)]))['compatible'])

    def test_disconnected_parts_are_insufficient(self):
        self.assertFalse(verify_h(draw([(20,10,20,100),(90,10,90,100),(40,55,70,55)]))['compatible'])

    def test_n_has_stems_but_no_shared_horizontal(self):
        self.assertFalse(verify_h(draw([(20,10,20,100),(90,10,90,100),(20,10,90,100)]))['compatible'])

    def test_extra_box_bars_are_not_explained_by_h(self):
        self.assertFalse(verify_h(draw([(20,10,20,100),(90,10,90,100),(20,55,90,55),
                                      (20,10,90,10),(20,100,90,100)]))['compatible'])

    def test_white_image_is_not_h(self):
        self.assertFalse(verify_h(draw([]))['compatible'])


if __name__=='__main__':
    unittest.main()
