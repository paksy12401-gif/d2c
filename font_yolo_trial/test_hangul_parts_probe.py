import unittest
import cv2
import numpy as np

from hangul_parts_probe import decompose,add_vertical_vowel,right_vertical
from apply_hangul_parts import proposed_changes


class HangulPartsTests(unittest.TestCase):
    def test_unicode_composition_preserves_consonants(self):
        for source,target in [('도','되'),('구','귀'),('그','긔')]:
            self.assertEqual(add_vertical_vowel(source),target)
            a,b=decompose(source),decompose(target)
            self.assertEqual((a[0],a[2]),(b[0],b[2]))
        self.assertIsNone(add_vertical_vowel('H'))
        self.assertIsNone(add_vertical_vowel('가'))

    def test_right_stem_and_horizontal_only_layout(self):
        im=np.full((120,120,3),255,np.uint8)
        cv2.rectangle(im,(10,10),(65,55),(0,0,0),2)
        cv2.line(im,(10,95),(65,95),(0,0,0),2)
        cv2.line(im,(38,65),(38,95),(0,0,0),2)
        self.assertFalse(right_vertical(im)['present'])
        cv2.line(im,(95,10),(95,105),(0,0,0),2)
        self.assertTrue(right_vertical(im)['present'])

    def test_stroke_and_multiple_ctc_candidates_are_both_required(self):
        def case(present=True,score=.1):
            return dict(text='도',geometry=[dict(index=0,height=h,present=present) for h in (40,56,80)],
                        char_candidates=[dict(height=h,raw={'rec_text':['도',[0,[],[],[],[.9],[[dict(text='되',score=score)]]]]}) for h in (40,56,80)])
        self.assertEqual(proposed_changes(case())[0]['after'],'되')
        self.assertEqual(proposed_changes(case(False)),[])
        self.assertEqual(proposed_changes(case(score=.001)),[])
        record=case();record['char_candidates']=record['char_candidates'][:1]
        self.assertEqual(proposed_changes(record),[])


if __name__=='__main__':unittest.main()
