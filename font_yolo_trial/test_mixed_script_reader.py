import unittest

from mixed_script_reader import expand_detection, korean_body, merge_korean


class MixedScriptChecks(unittest.TestCase):
    def test_partial_detection_expands_to_full_row_and_stays_above_next_row(self):
        rows=[{'center':40,'top':20,'bottom':60},{'center':100,'top':80,'bottom':120}]
        result=expand_detection([30,20,120,40],rows,200,150)
        self.assertLessEqual(result['xyxy'][1],20)
        self.assertGreaterEqual(result['xyxy'][3],60)
        self.assertLessEqual(result['xyxy'][3],70)

    def test_only_confident_hangul_is_accepted_without_correcting_words(self):
        self.assertEqual(korean_body('적용할것.',.99),'적용할것')
        self.assertIsNone(korean_body('적용할것',.6))
        self.assertIsNone(korean_body('SLEEVE 적용할것',.99))

    def test_korean_replaces_internal_letters_and_preserves_symbol_and_outer_letters(self):
        boxes=[[0,0,10,20,.9,15],[30,0,40,20,.4,5],[49,15,51,20,.9,65]]
        recognized=[{'xyxy':[25,0,55,20],'korean_text':'적용','score':.99}]
        merged,names,suppressed=merge_korean(boxes,{i:str(i) for i in range(67)},recognized)
        self.assertEqual(set(merged[:,5].astype(int)),{15,65,67})
        self.assertEqual(names[67],'적용')
        self.assertEqual(len(suppressed),1)


if __name__=='__main__':unittest.main()
