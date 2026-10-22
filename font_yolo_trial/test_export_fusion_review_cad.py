import unittest
from export_fusion_review_cad import StrokeIndex


class StrokeCorrespondenceTests(unittest.TestCase):
    def test_only_a_complete_corresponding_entity_moves(self):
        index=StrokeIndex({'A':[(0,0,1,0),(1,0,1,1)]})
        self.assertEqual(index.match([(0,0),(1,0),(1,1)])[0],'A')
        self.assertIsNone(index.match([(0,0),(1,0),(1,2)])[0])

    def test_small_import_rounding_is_tolerated(self):
        index=StrokeIndex({'A':[(0,0,1,0)]})
        self.assertEqual(index.match([(.01,0),(1.01,0)])[0],'A')
        self.assertIsNone(index.match([(.1,0),(1.1,0)])[0])

    def test_a_wire_joined_to_matching_text_is_preserved(self):
        index=StrokeIndex({'A':[(0,0,1,0)],'B':[(1,0,2,0)]})
        self.assertIsNone(index.match([(0,0),(1,0),(2,0)])[0])
        self.assertIsNone(index.match([(0,0),(5,0)])[0])


if __name__=='__main__':unittest.main()
