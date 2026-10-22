import unittest

import numpy as np

from prepare import yolo_labels
from train_vertical_bar import NAMES, extract_bars, scene
from vertical_bar_reader import merge_bars


class VerticalBarTests(unittest.TestCase):
    def test_case_confusion_can_support_height_but_shorter_lowercase_cannot(self):
        image=np.full((140,190,3),255,np.uint8)
        image[30:80,20:40]=0;image[30:80,130:150]=0;image[10:80,85:90]=0
        old=[[20,30,40,80,.99,0],[130,30,150,80,.99,NAMES.index('x')]]
        candidate=[[83,9,92,81,.99,67]]
        _,detail=merge_bars(image,old,candidate,min_height_ratio=1.15)
        self.assertEqual(len(detail['accepted']),1)
        image[30:50,130:150]=255;old[1][1]=50
        _,detail=merge_bars(image,old,candidate,min_height_ratio=1.15)
        self.assertEqual(detail['height_checks'][0]['reason'],'no_reference')

    def test_actual_height_rejects_I_even_with_tall_predicted_box(self):
        image=np.full((140,190,3),255,np.uint8)
        image[30:80,20:40]=0;image[30:80,130:150]=0
        image[30:80,85:90]=0
        old=[[20,30,40,80,.99,0],[130,30,150,80,.99,7],[85,30,90,80,.95,8]]
        candidate=[[83,10,92,95,.99,67]]
        result,detail=merge_bars(image,old,candidate,min_height_ratio=1.15)
        self.assertEqual([int(b[5]) for b in result],[0,7,8])
        self.assertEqual(detail['height_checks'][0]['height_ratio'],1)
        self.assertEqual(detail['height_checks'][0]['reason'],'not_longer')

    def test_taller_stroke_accepted_and_rule_is_scale_invariant(self):
        import cv2
        image=np.full((140,190,3),255,np.uint8)
        image[30:80,20:40]=0;image[30:80,130:150]=0
        image[10:80,85:90]=0
        old=[[20,30,40,80,.99,0],[130,30,150,80,.99,7],[85,10,90,80,.8,37]]
        candidate=[[83,9,92,81,.99,67]]
        for scale in [1,2]:
            im=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            scaled=lambda rows:[[v*scale if j<4 else v for j,v in enumerate(b)] for b in rows]
            result,detail=merge_bars(im,scaled(old),scaled(candidate),min_height_ratio=1.15)
            self.assertIn(67,[int(b[5]) for b in result])
            self.assertAlmostEqual(detail['height_checks'][0]['height_ratio'],1.4)
        _,detail=merge_bars(image,old[2:],candidate,min_height_ratio=1.15)
        self.assertEqual(detail['height_checks'][0]['reason'],'no_reference')
        with self.assertRaises(ValueError):merge_bars(image,old,candidate,min_height_ratio=1)

    def test_ambiguous_stem_inside_word_is_withheld(self):
        image=np.full((140,180,3),255,np.uint8)
        image[30:100,50:58]=0
        old=[[15,30,43,100,.98,0],[50,30,58,100,.3,37],[66,30,100,100,.99,1]]
        new=[[49,29,59,101,.95,67]]
        result,detail=merge_bars(image,old,new,protect_words=True)
        self.assertEqual(result.shape[0],3)
        self.assertEqual(len(detail['accepted']),0)
        self.assertEqual(len(detail['context_rejected']),1)

    def test_cursor_excluded_and_three_bars_share_one_class(self):
        image = np.full((185, 207, 3), 32, np.uint8)
        image[:34, 29] = 180
        for x, bottom in [(27, 137), (64, 113), (103, 113)]:
            image[34:bottom, x:x+10] = 255
        bars, boxes = extract_bars(image)
        self.assertEqual(len(bars), 3)
        self.assertEqual([b[1] for b in boxes], [34]*3)
        self.assertEqual(NAMES[67], '|')
        self.assertEqual([NAMES[i] for i in (8, 37, 53)], ['I', 'l', '1'])
        self.assertTrue(all(b['char'] == '|' for b in bars))

    def test_scene_labels_stay_inside_canvas_and_include_bar(self):
        alpha = np.ones((25, 10), np.float32)
        glyphs = [dict(alpha=alpha, char=c, reference_height=25, top_from_baseline=-25) for c in NAMES[:67]]
        bars = [dict(alpha=np.ones((27, 5), np.float32), char='|', reference_height=25, top_from_baseline=-26)]
        seen = 0
        for seed in range(60):
            image, objects = scene(glyphs, bars, seed)
            labels = yolo_labels(objects, 384, 384, 68)
            self.assertEqual(len(labels.splitlines()), len(objects))
            seen += sum(o['class_id'] == 67 for o in objects)
        self.assertGreater(seen, 60)
        image, objects = scene(glyphs, bars, 1, negative=True)
        self.assertEqual(objects, [])

    def test_merge_replaces_plain_stem_but_preserves_I_and_long_line(self):
        image = np.full((160, 200, 3), 255, np.uint8)
        image[30:100, 30:38] = 0
        image[30:100, 80:84] = 0
        image[30:34, 70:94] = 0
        image[96:100, 70:94] = 0
        image[5:155, 140:143] = 0
        old = [[28,28,40,102,.9,37], [68,28,96,102,.99,8]]
        candidates = [[29,29,39,101,.95,67], [79,29,85,101,.9,67], [139,30,144,100,.95,67]]
        boxes, details = merge_bars(image, old, candidates)
        self.assertEqual(sorted(int(b[5]) for b in boxes), [8,67])
        self.assertEqual(len(details['accepted']), 1)
        self.assertEqual(len(details['rejected']), 2)


if __name__ == '__main__':
    unittest.main()
