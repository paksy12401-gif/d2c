import unittest
import numpy as np

from iterate_fusion import accepted_korean, fuse_words, lines, word_regions, field_box, owned_by_field, vector_symbol, hybrid_glyphs
from train_spacing_trial import NAMES, scene


class FusionIterationChecks(unittest.TestCase):
    def test_mixed_digit_hangul_is_kept_but_english_is_not_overwritten(self):
        self.assertEqual(accepted_korean('2회',.95),'2회')
        self.assertIsNone(accepted_korean('2회',.85))
        self.assertIsNone(accepted_korean('LIST 참조',.99))

    def test_ocr_suppression_requires_coverage_not_only_center(self):
        glyphs=[dict(text='A',score=.95,box_pdf=[0,0,20,20]),dict(text='F',score=.6,box_pdf=[8,8,12,12])]
        words=[dict(accepted='적용',score=.99,box_pdf=[7,7,13,13])]
        kept,suppressed=fuse_words(glyphs,words)
        self.assertEqual([g['text'] for g in suppressed],['F'])
        self.assertIn('A',[g['text'] for g in kept])

    def test_full_field_extent_keeps_original_lower_punctuation(self):
        self.assertEqual(field_box(dict(box_pdf=[0,2,20,12],source_box_before_height_measurement=[0,0,20,15])),[0,0,20,15])

    def test_baseline_symbol_does_not_become_a_second_row(self):
        glyphs=[dict(text='A',score=.9,box_pdf=[0,0,8,10]),dict(text='_',score=1,box_pdf=[8,11,18,11]),dict(text='B',score=.9,box_pdf=[18,0,26,10])]
        self.assertEqual(''.join(''.join(lines(glyphs)).split()),'A_B')

    def test_word_split_does_not_depend_on_recognized_language(self):
        im=np.full((40,120,3),255,np.uint8);im[5:30,5:20]=0;im[5:30,60:80]=0
        self.assertEqual(len(word_regions(im,25)),2)

    def test_negative_training_scene_has_no_character_labels(self):
        im,objects=scene([],123,blank=True)
        self.assertEqual(objects,[])
        self.assertTrue((im<128).any())
        self.assertEqual(len(NAMES),68)

    def test_partial_character_from_neighbor_field_is_not_owned(self):
        field=dict(box_pdf=[10,0,30,12],cap=10)
        self.assertFalse(owned_by_field(dict(box_pdf=[8,11,15,11]),field))
        self.assertTrue(owned_by_field(dict(box_pdf=[11,0,20,10]),field))

    def test_ocr_omission_does_not_delete_yolo_period(self):
        glyphs=[dict(text='.',score=.9,box_pdf=[9,8,10,9])]
        kept,_=fuse_words(glyphs,[dict(accepted='적용',score=.99,box_pdf=[0,0,10,10])])
        self.assertIn('.',[g['text'] for g in kept])

    def test_small_disconnected_korean_stem_stays_in_word(self):
        im=np.full((70,130,3),255,np.uint8)
        im[10:60,5:50]=0;im[10:60,72:74]=0
        self.assertEqual(len(word_regions(im,56)),1)

    def test_vector_period_and_colon_are_geometry_rules(self):
        def square(x,y):return [(x,y,x+1,y),(x+1,y,x+1,y+1),(x+1,y+1,x,y+1),(x,y+1,x,y)]
        colon=dict(box=[0,1,1,7],segments=square(0,1)+square(0,6))
        self.assertEqual(vector_symbol(colon,10,10),':')
        self.assertEqual(vector_symbol(dict(box=[0,9,1,10],segments=square(0,9)),10,10),'.')

    def test_symbol_supplement_cannot_replace_full_height_p(self):
        glyph=dict(text='P',score=.8,box_pdf=[0,0,10,10])
        false_symbol=dict(text=':',score=.99,box_pdf=[2,2,5,8])
        self.assertEqual(hybrid_glyphs([glyph],[false_symbol],10),[glyph])

    def test_missing_external_period_can_be_supplemented(self):
        glyph=dict(text='P',score=.8,box_pdf=[0,0,10,10])
        period=dict(text='.',score=.9,box_pdf=[12,8,14,10])
        self.assertEqual([g['text'] for g in hybrid_glyphs([glyph],[period],10)],['P','.'])


if __name__=='__main__':unittest.main()
