"""Regression cases for preserving letters while resolving line-shaped symbol conflicts."""
import unittest

import cv2
import numpy as np

from preserved_reader import merge_predictions, remove_drawing_rules, symbol_nms


class PreservationChecks(unittest.TestCase):
    def test_rule_is_removed_but_a_connected_text_baseline_is_kept(self):
        image = np.full((130, 300, 3), 255, dtype=np.uint8)
        cv2.line(image, (10, 50), (280, 50), (0, 0, 0), 1)
        cv2.line(image, (10, 75), (280, 75), (0, 0, 0), 1)
        boxes = [[10, 10, 100, 51, .9, 0], [105, 10, 190, 51, .9, 0],
                 [195, 10, 281, 51, .9, 0]]
        clean, mask, _ = remove_drawing_rules(image, boxes)
        self.assertTrue(np.array_equal(clean[50], image[50]))
        self.assertTrue(np.all(clean[75] == 255))
        self.assertGreater(mask[75].sum(), 0)
        self.assertEqual(mask[50].sum(), 0)

    def test_real_letter_survives_an_inner_horizontal_symbol_candidate(self):
        image = np.full((70, 80, 3), 255, dtype=np.uint8)
        cv2.line(image, (20, 12), (20, 55), (0, 0, 0), 2)
        cv2.line(image, (20, 12), (48, 12), (0, 0, 0), 2)
        cv2.line(image, (20, 32), (48, 32), (0, 0, 0), 2)
        letter = [18, 10, 51, 58, .9, 5]
        symbol = [20, 29, 49, 35, .9, 63]
        merged, replaced = merge_predictions([letter], [symbol], image)
        self.assertEqual(replaced, [])
        self.assertTrue(any(int(box[5]) == 5 for box in merged))

    def test_letter_shaped_false_positive_is_replaced_when_only_a_dash_remains(self):
        image = np.full((70, 80, 3), 255, dtype=np.uint8)
        cv2.line(image, (20, 32), (49, 32), (0, 0, 0), 1)
        letter = [18, 10, 51, 58, .9, 5]
        symbol = [19, 30, 51, 35, .8, 63]
        merged, replaced = merge_predictions([letter], [symbol], image)
        self.assertEqual(len(replaced), 1)
        self.assertEqual(merged[:, 5].tolist(), [63])

    def test_duplicate_symbol_is_removed_without_removing_an_adjacent_one(self):
        boxes = [[20, 20, 30, 30, .9, 65], [21, 20, 31, 30, .2, 65],
                 [35, 20, 45, 30, .8, 65]]
        kept = symbol_nms(boxes)
        self.assertEqual(len(kept), 2)
        self.assertEqual([box[4] for box in kept], [.9, .8])

    def test_confident_l_is_not_overwritten_by_a_colon_at_the_same_position(self):
        image = np.full((70, 80, 3), 255, dtype=np.uint8)
        cv2.line(image, (30, 12), (30, 55), (0, 0, 0), 1)
        letter = [28, 10, 33, 58, .82, 37]
        symbol = [28, 10, 33, 58, .9, 62]
        merged, replaced = merge_predictions([letter], [symbol], image)
        self.assertEqual(replaced, [])
        self.assertTrue(any(int(box[5]) == 37 for box in merged))


if __name__ == "__main__":
    unittest.main()
