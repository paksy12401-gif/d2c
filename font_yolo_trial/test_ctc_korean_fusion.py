import unittest
import numpy as np

from ctc_korean_fusion import ctc_confidences,korean_spans,choose_spans
from read_frame_labels import choose_label,path_bounds
import pymupdf as fitz
from vector_extra_symbols import extra_symbol


class CtcFusionChecks(unittest.TestCase):
    def test_ctc_collapse_tracks_repeats_and_blank_separated_repeats(self):
        ids=[0,1,1,0,1,2,0]
        data=np.zeros((1,len(ids),3));data[0,np.arange(len(ids)),ids]=np.arange(len(ids))*.01+.90
        self.assertTrue(np.allclose(ctc_confidences(data)[0],[.91,.94,.95]))

    def test_korean_only_positions_preserve_the_english_prefix(self):
        raw={'rec_text':['AB 적용',[20,[['A','B'],[' ','적','용']],[[1,3],[5,10,14]],['en&num','symbol'],[.99]*5]]}
        spans=korean_spans(raw,200,[0,0],2,10,[0,0,100,10])
        self.assertEqual(spans[0]['accepted'],'적용')
        self.assertGreater(spans[0]['box_pdf'][0],30)
        self.assertLess(spans[0]['box_pdf'][2],100)

    def test_interval_selection_does_not_duplicate_the_same_word(self):
        candidates=[dict(accepted='적용',text='적용',score=.98,box_pdf=[0,0,20,10],variant='a'),dict(accepted='적용',text='적용',score=.99,box_pdf=[1,0,21,10],variant='b'),dict(accepted='참조',text='참조',score=.99,box_pdf=[30,0,50,10],variant='a')]
        selected=choose_spans(candidates,10)
        self.assertEqual([s['accepted'] for s in selected],['적용','참조'])
        self.assertEqual(len(selected[0]['variants']),2)

    def test_previous_accepted_word_is_not_replaced_by_conflicting_retry(self):
        baseline=dict(accepted='적용',text='적용',score=.94,box_pdf=[0,0,20,10],variant='previous_word')
        retry=dict(accepted='작용',text='작용',score=.99,box_pdf=[0,0,20,10],variant='retry')
        self.assertEqual(choose_spans([baseline,retry],10)[0]['accepted'],'적용')

    def test_complete_ink_box_is_kept_when_same_text_has_a_narrow_ctc_span(self):
        baseline=dict(accepted='적용',text='적용',score=.94,box_pdf=[0,0,20,10],variant='previous_word')
        retry=dict(accepted='적용',text='적용',score=.99,box_pdf=[1,0,18,10],variant='retry')
        self.assertEqual(choose_spans([baseline,retry],10)[0]['box_pdf'],[0,0,20,10])

    def test_incomplete_pdf_rect_is_rebuilt_from_actual_endpoints(self):
        path=dict(rect=fitz.Rect(0,4,8,10),items=[('l',fitz.Point(0,4),fitz.Point(8,4)),('l',fitz.Point(0,0),fitz.Point(0,10))])
        self.assertEqual(list(path_bounds(path)),[0,0,8,10])

    def test_frame_context_never_fills_an_expected_sequence(self):
        self.assertEqual(choose_label('O2',.9,'02',.99,'numeric')[0],'02')
        self.assertEqual(choose_label('B',.96,'8',.99,'letter')[0],'B')
        self.assertEqual(choose_label('?',.1,'?',.1,'numeric')[0],'?')

    def test_parenthesis_and_midline_dash_geometry(self):
        left=dict(box=[0,0,2,10],segments=[(2,0,0,5),(0,5,2,10)])
        right=dict(box=[0,0,2,10],segments=[(0,0,2,5),(2,5,0,10)])
        self.assertEqual(extra_symbol(left,10,10),'(')
        self.assertEqual(extra_symbol(right,10,10),')')
        self.assertEqual(extra_symbol(dict(box=[0,5,10,5],segments=[(0,5,10,5)]),10,10),'-')
        self.assertIsNone(extra_symbol(dict(box=[0,0,2,10],segments=[(0,0,0,10),(0,10,2,10)]),10,10))


if __name__=='__main__':unittest.main()
