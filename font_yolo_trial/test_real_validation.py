import unittest
from evaluate_real_validation import alignment, supported_score
from infer_real_validation import inside


class RealValidationChecks(unittest.TestCase):
    def test_alignment_reports_omitted_and_extra_characters(self):
        score=alignment('ABC','ACX')
        self.assertEqual(score['edit_distance'],2)
        self.assertEqual(sum(score[k] for k in ['substitutions','deletions','insertions']),2)
        self.assertEqual(alignment('FLEECE TAPE','FLEECETAPE')['edit_distance'],0)

    def test_o_zero_ambiguity_is_local_and_does_not_correct_output(self):
        self.assertEqual(alignment('A470','A47O',[{'index':3,'allowed':['0','O']}])['edit_distance'],0)
        self.assertEqual(alignment('FRONT','FR0NT')['edit_distance'],1)

    def test_untrained_separator_is_reported_separately(self):
        answer={'reference':'FRONT_X1','ambiguities':[]}
        self.assertEqual(alignment(answer['reference'],'FRONTX1')['edit_distance'],1)
        self.assertEqual(supported_score(answer,'FRONTX1',set('FRONTX1'))['edit_distance'],0)

    def test_drawing_negatives_count_all_inserted_predictions(self):
        self.assertEqual(alignment('',':00')['insertions'],3)
        self.assertIsNone(alignment('',':00')['character_error_rate'])

    def test_center_selection_and_touching_edges(self):
        self.assertTrue(inside([1,1,3,3],[0,0,4,4]))
        self.assertFalse(inside([5,5,7,7],[0,0,4,4]))


if __name__=='__main__':unittest.main()
