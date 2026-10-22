import unittest
from non_table_cad import in_table, prediction_layer


class ConversionChecks(unittest.TestCase):
    def test_tables_are_excluded_and_work_notes_and_frame_are_kept(self):
        self.assertTrue(in_table([5100,500,5120,520]))
        self.assertTrue(in_table([8011,95,11150,2209]))
        self.assertFalse(in_table([6060,1300,7140,1590]))
        self.assertFalse(in_table([0,0,12500,2383]))

    def test_yolo_predictions_are_visible_despite_disagreeing_or_empty_ocr(self):
        for label in ('P','0',':','-','/'):
            item={'kind':'yolo','text':label,'score':.1}
            self.assertEqual(prediction_layer(item,{'paddle_raw':'F','paddle_score':.99}),'YOLO_TEXT')
            self.assertEqual(prediction_layer(item,{'paddle_raw':'','paddle_score':0}),'YOLO_TEXT')

    def test_korean_acceptance_is_unchanged(self):
        self.assertEqual(prediction_layer({'kind':'korean','score':.99},{'paddle_score':.9}),'OCR_TEXT')
        self.assertEqual(prediction_layer({'kind':'korean','score':.8},{'paddle_score':.9}),'OCR_REVIEW')


if __name__=='__main__':unittest.main()
