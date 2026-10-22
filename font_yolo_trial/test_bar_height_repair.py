import unittest

from repair_bar_height import vertical_evidence


class VectorBarHeightTests(unittest.TestCase):
    def test_short_I_preserved_tall_continuous_bar_accepted(self):
        self.assertFalse(vertical_evidence([10,0,10,10],10,[(10,0,10,10)])['accepted'])
        self.assertTrue(vertical_evidence([10,0,10,14],10,[(10,0,10,7),(10,7,10,14)])['accepted'])

    def test_dotted_or_wide_stroke_is_not_a_plain_bar(self):
        self.assertFalse(vertical_evidence([10,0,10,14],10,[(10,0,10,2),(10,5,10,14)])['accepted'])
        self.assertFalse(vertical_evidence([10,0,13,14],10,[(10,0,10,14)])['accepted'])


if __name__=='__main__':unittest.main()
