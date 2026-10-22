import unittest

import numpy as np

from refine_table_ocr import alphabet, character_vote, color_tokens, color_reread, connector_match, ctc_details


class TableRefinementTests(unittest.TestCase):
    def test_numeric_constraints_preserve_mixed_pin_ids_and_headers(self):
        self.assertIsNone(alphabet(dict(table='right',row=1,column=8)))
        self.assertIsNone(alphabet(dict(table='right',row=0,column=0)))
        self.assertIsNone(alphabet(dict(table='left',row=0,column=3)))
        self.assertIn('-',alphabet(dict(table='right',row=1,column=2)))

    def test_connector_crosscheck_rejects_conflicting_sources_and_other_ids(self):
        self.assertEqual(connector_match('12345678_A0_X1',{'12345678_AO_X1'}),'12345678_AO_X1')
        self.assertIsNone(connector_match('12345678_A0_X1',{'87654321_AO_X1'}))
        self.assertIsNone(connector_match('12345678_A0_X1',{'12345678_A0_X1','12345678_AO_X1'}))
        self.assertIsNone(connector_match('12345678_A7_X1',{'12345678_AO_X1'}))

    def test_votes_require_support_and_do_not_change_unrelated_digits(self):
        def view(text):return dict(ctc=dict(chars=[dict(text=c,score=.99) for c in text]))
        value,changes=character_vote('A 0',{'one':view('AO'),'two':view('AO')},[])
        self.assertEqual(value,'A 0');self.assertEqual(changes,[])
        value,changes=character_vote('A 0',{'one':view('AO'),'two':view('AO'),'three':view('AO')},[])
        self.assertEqual(value,'A O');self.assertEqual(len(changes),1)
        value,_=character_vote('A1',{'one':view('A7'),'two':view('A7'),'three':view('A7')},[])
        self.assertEqual(value,'A1')
        value,_=character_vote('A00',{'one':view('AOO'),'two':view('AOO'),'three':view('AOO')},[])
        self.assertEqual(value,'A00')

    def test_color_constraint_requires_visual_alternative(self):
        def view(text, alternative):
            return dict(text=text,score=.96,ctc=dict(chars=[dict(text=c,score=.8,alternatives=[dict(text='V',score=alternative)]) for c in text]))
        self.assertEqual(color_reread('BU/NT',[view('BU/NT',.35),view('BU/VT',.8)],{'BU':3,'VT':3}),'BU/VT')
        self.assertIsNone(color_reread('BU/NT',[view('BU/NT',.001),view('BU/VT',.8)],{'BU':3,'VT':3}))
        self.assertIsNone(color_reread('BU/NT',[view('BU/NT',.35),view('BU/VT',.8)],{'BU':3}))

    def test_color_vocabulary_requires_two_input_agreement(self):
        rows=[]
        for variant,text,score in [('raw_cell','VT/BK',.99),('clean_cell','VT/BK',.99)]:
            rows.append(dict(id='a',variant=variant,table='right',row=1,column=3,text=text,score=score))
        for variant,text,score in [('raw_cell','WH/VT',.99),('clean_cell','WH/WT',.84)]:
            rows.append(dict(id='b',variant=variant,table='right',row=2,column=3,text=text,score=score))
        tokens,evidence=color_tokens(rows)
        self.assertEqual(set(tokens),{'VT','BK'});self.assertEqual(len(evidence),1)

    def test_ctc_retains_repetition_separated_by_blank(self):
        indexes=[0,1,1,0,1,2,2]
        probabilities=np.full((len(indexes),3),.01)
        for row,index in enumerate(indexes):probabilities[row,index]=.98
        decoded=ctc_details(probabilities,['blank','A','B'])
        self.assertEqual(decoded['text'],'AAB')
        self.assertEqual([c['column'] for c in decoded['chars']],[1,4,5])


if __name__=='__main__':unittest.main()
