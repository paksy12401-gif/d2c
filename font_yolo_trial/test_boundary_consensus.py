import unittest

from boundary_consensus import easyocr_content_columns, spatial_consensus


class BoundaryTests(unittest.TestCase):
    def test_padding_mapping(self):
        columns = easyocr_content_columns(31, (94, 116, 3))
        self.assertAlmostEqual(columns, 31*79/128)
        self.assertAlmostEqual(6.5/columns*116, 6.5/31*128/79*116)
        self.assertEqual(easyocr_content_columns(31, (64, 128)), 31)

    def test_high_text_score_does_not_choose_outlier_box(self):
        def p(variant, left, score=.9):
            return dict(variant=variant, text='X', accepted='X', score=score,
                        box_pdf=[left, 0, left+10, 10])
        proposals = [p('vector_h40', 20), p('native_h40', 20.2),
                     p('ink_h40', 19.8), p('easy_word0_s1', 16, .999)]
        result = spatial_consensus(proposals, 10)[0]
        self.assertTrue(result['boundary_stable'])
        self.assertEqual(result['box_pdf'], [20, 0, 30, 10])
        self.assertEqual(result['box_pdf'], spatial_consensus(proposals+proposals[-1:]*100, 10)[0]['box_pdf'])

    def test_one_reader_family_cannot_confirm_boundary(self):
        ps = [dict(variant='easy_word0_s'+str(i), accepted='X', score=.99,
                   box_pdf=[10, 0, 20, 10]) for i in range(5)]
        self.assertFalse(spatial_consensus(ps, 10)[0]['boundary_stable'])

    def test_separated_occurrences_are_not_merged(self):
        ps = [dict(variant='vector_h40', accepted='X', score=.99,
                   box_pdf=[x, 0, x+10, 10]) for x in (0, 20)]
        self.assertEqual(len(spatial_consensus(ps, 10)), 2)


if __name__ == '__main__':
    unittest.main()
