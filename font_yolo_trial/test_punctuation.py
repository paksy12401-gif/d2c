"""Checks for case-sensitive transfer and baseline-relative punctuation handling."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from prepare import PUNCTUATION, extract_glyphs, resize_glyph, paste_glyph
from train import copy_classifier_rows, read_rows


class Boxes:
    def __init__(self, rows):
        self.data = np.asarray(rows, dtype=float).reshape(-1, 6)

    def cpu(self):
        return self

    def numpy(self):
        return self


class PunctuationChecks(unittest.TestCase):
    def test_thin_hyphen_survives_training_filter_without_thickening_pixels(self):
        from ultralytics.data.augment import RandomPerspective
        image = np.full((80, 100, 3), 255, dtype=np.uint8)
        alpha = np.ones((2, 30), dtype=np.float32)
        item = paste_glyph(image, alpha, 20, 30, np.zeros(3), 63, 0, "-")
        box = np.asarray(item["bbox_xyxy"], dtype=float).reshape(4, 1)
        self.assertTrue(RandomPerspective.box_candidates(box, box)[0])
        self.assertEqual(np.flatnonzero((image < 128).any(axis=(1, 2))).tolist(), [30, 31])

    def test_upper_and_lower_case_transfer_to_different_rows(self):
        old = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1))
        new = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1))
        with torch.no_grad():
            for i in range(3):
                old[-1].weight[i].fill_(i + 1)
                old[-1].bias[i].fill_(i + 10)
        new_weight, new_bias = new[-1].weight[3].clone(), new[-1].bias[3].clone()
        source = SimpleNamespace(names={0: "A", 1: "a", 2: "0"}, model=[SimpleNamespace(cv3=[old])])
        target = SimpleNamespace(model=[SimpleNamespace(cv3=[new])])
        report = copy_classifier_rows(source, target, {0: "a", 1: "A", 2: "0", 3: ":"})
        self.assertEqual(report["matched_classes"], 3)
        self.assertTrue(torch.equal(new[-1].weight[0], old[-1].weight[1]))
        self.assertTrue(torch.equal(new[-1].weight[1], old[-1].weight[0]))
        self.assertTrue(torch.equal(new[-1].weight[3], new_weight))
        self.assertTrue(torch.equal(new[-1].bias[3], new_bias))

    def test_small_marks_stay_on_their_text_row(self):
        names = {0: "A", 1: ":", 2: "-", 3: "'", 4: ".", 5: "B"}
        boxes = Boxes([[20, 20, 40, 60, .9, 0], [44, 30, 48, 57, .9, 1],
                       [50, 38, 66, 40, .9, 2], [68, 20, 71, 30, .9, 3],
                       [75, 55, 79, 60, .9, 4], [20, 90, 40, 130, .9, 5],
                       [43, 125, 47, 130, .9, 4]])
        self.assertEqual(read_rows(SimpleNamespace(boxes=boxes, names=names)), ["A:-'.", "B."])
        self.assertEqual(read_rows(SimpleNamespace(boxes=Boxes([]), names=names)), [])

    def test_supplied_colon_keeps_both_dots_and_period_stays_small(self):
        source = Path(__file__).resolve().parent / "punctuation/source_punctuation.png"
        scratch = source.parent / "checks"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(scratch.resolve()))
            glyphs, boxes = extract_glyphs(source, Path(directory), PUNCTUATION, 62, reference_char="/")
            self.assertEqual([box["char"] for box in boxes], list(PUNCTUATION))
            colon = glyphs[0]["alpha"]
            self.assertGreater(colon[:colon.shape[0]//3].max(), .5)
            self.assertGreater(colon[-colon.shape[0]//3:].max(), .5)
            period, _ = resize_glyph(glyphs[3], 48, 1)
            slash, _ = resize_glyph(glyphs[2], 48, 1)
            self.assertLess(period.shape[0], slash.shape[0] / 3)
            self.assertEqual(len(list((Path(directory) / "glyphs").glob("*.png"))), 5)


if __name__ == "__main__":
    unittest.main()
