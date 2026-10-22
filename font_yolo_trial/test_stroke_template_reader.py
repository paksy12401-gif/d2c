import unittest

import cv2
import numpy as np

from stroke_template_reader import StrokeReader,stroke_features


def shape(segments):
    im=np.full((120,120,3),255,np.uint8)
    for a,b,c,d in segments:cv2.line(im,(a,b),(c,d),(0,0,0),2)
    return im


class StrokeTemplateTests(unittest.TestCase):
    def test_same_stems_different_crossbar_direction_are_distinguished(self):
        h=shape([(20,10,20,100),(90,10,90,100),(20,55,90,55)])
        n=shape([(20,10,20,100),(90,10,90,100),(20,10,90,100)])
        reader=StrokeReader({'H':h,'N':n})
        for char,im in [('H',h),('N',n)]:
            result=reader.predict(im,proposed=char)
            self.assertEqual(result['text'],char)
            self.assertLess(result['distance'],.01)
            self.assertGreater(result['margin'],0)

    def test_extra_strokes_are_penalized_in_both_directions(self):
        f=shape([(20,10,20,100),(20,10,90,10),(20,55,90,55)])
        e=f.copy();cv2.line(e,(20,100),(90,100),(0,0,0),2)
        reader=StrokeReader({'F':f,'E':e})
        self.assertEqual(reader.predict(e,proposed='F')['text'],'E')
        self.assertGreater(reader.predict(e,proposed='F')['proposed_distance'],.5)

    def test_blank_image_has_no_identification(self):
        im=shape([(20,10,20,100)])
        self.assertEqual(StrokeReader({'l':im,'I':im}).predict(shape([]))['text'],'')

    def test_stroke_points_and_direction_channels_are_aligned(self):
        f=stroke_features(shape([(20,10,20,100),(20,55,90,55)]))
        self.assertEqual(len(f['ys']),len(f['channels']))
        self.assertEqual(f['distance'].shape,(4,112,112))


if __name__=='__main__':unittest.main()
