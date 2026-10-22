import json
import unittest
from pathlib import Path

import numpy as np

from train_table_font_verifier import OUT,patch,target_class,verified_character
from combine_table_font_server import contextual_character


class FontVerifierTests(unittest.TestCase):
    def test_indistinguishable_stems_share_training_class(self):
        self.assertEqual(target_class('I'),target_class('l'))
        self.assertNotEqual(target_class('I'),target_class('1'))

    def test_confidence_and_shift_agreement_are_both_required(self):
        self.assertEqual(verified_character('O','0',.999,True,[]),'0')
        self.assertEqual(verified_character('O','0',.98,True,[]),'O')
        self.assertEqual(verified_character('O','0',.999,False,[]),'O')
        self.assertEqual(verified_character('O','1',.999,True,[]),'O')

    def test_ambiguous_stem_requires_context_to_choose_case(self):
        self.assertEqual(verified_character('|','Il',.999,True,[]),'|')
        self.assertEqual(verified_character('|','Il',.999,True,['I','l']),'|')
        self.assertEqual(verified_character('|','Il',.999,True,['I','I']),'I')
        self.assertEqual(verified_character('I','Il',.999,True,['l','l']),'l')

    def test_patch_excludes_neighbor_stroke(self):
        image=np.full((41,81),255,np.uint8);image[:,20]=0;image[:,24]=0
        original=image.copy()
        value=patch(image,dict(bbox_pdf=[0,0,20,10]),dict(box_pdf=[5,0,5,10]),10)
        yy,xx=np.where(value<100)
        self.assertGreater(len(xx),20)
        self.assertLessEqual(int(xx.max()-xx.min()),2)
        self.assertTrue(np.array_equal(image,original))

    def test_training_and_holdout_exclude_duplicate_whole_strings(self):
        train=json.loads((OUT/'train.json').read_text(encoding='utf-8'))['cells']
        held=json.loads((OUT/'holdout.json').read_text(encoding='utf-8'))['answers']
        self.assertFalse({r['id'] for r in train}&{r['id'] for r in held})
        self.assertFalse({''.join(r['reference'].split()) for r in train}&{''.join(r['reference'].split()) for r in held})

    def test_server_context_needs_shape_and_two_agreeing_views(self):
        shape=dict(shape_class='Il',shift_agreement=True,min_shift_confidence=.999)
        self.assertEqual(contextual_character('|','I','I',shape,1,.99),'I')
        self.assertEqual(contextual_character('|','I','l',shape,1,.99),'|')
        self.assertEqual(contextual_character('1','I','I',dict(shape_class='1'),1,.99),'1')
        self.assertEqual(contextual_character('O','0','0',None,1,.99),'O')

    def test_case_change_needs_corresponding_source_height(self):
        self.assertEqual(contextual_character('s','S','S',None,1,.99),'S')
        self.assertEqual(contextual_character('s','S','S',None,.75,.99),'s')
        self.assertEqual(contextual_character('S','s','s',None,1,.99),'S')


if __name__=='__main__':unittest.main()
