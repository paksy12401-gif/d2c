"""End-to-end local checks for the one-click revision pipeline."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from unified_revision import local_preliminary_review, run_local_analysis


def drawing_bytes(extra=False):
    image = np.full((500, 800, 3), 255, np.uint8)
    for x in range(40, 761, 80):
        cv2.line(image, (x, 30), (x, 470), (0, 0, 0), 1)
    for y in range(30, 471, 55):
        cv2.line(image, (40, y), (760, y), (0, 0, 0), 1)
    for index, x in enumerate(range(80, 721, 100)):
        cv2.circle(image, (x, 160 + index % 2 * 120), 18, (0, 0, 0), 2)
    cv2.putText(image, "100V", (300, 245), cv2.FONT_HERSHEY_SIMPLEX,
                .7, (0, 0, 0), 2, cv2.LINE_AA)
    if extra:
        cv2.line(image, (285, 330), (515, 330), (0, 0, 0), 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class FakeOCR:
    _cad_model_root = None

    def __init__(self):
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        text = "100V" if self.calls == 1 else "120V"
        return [SimpleNamespace(json={"res": {
            "rec_texts": [text], "rec_scores": [.96],
            "rec_polys": [[[295, 220], [365, 220], [365, 250], [295, 250]]],
        }})]


class UnifiedRevisionChecks(unittest.TestCase):
    def test_one_call_combines_visual_ocr_graph_and_review(self):
        result = run_local_analysis(
            drawing_bytes(), drawing_bytes(extra=True), "png", "png",
            dpi=100, sensitivity="high", ocr_reader=FakeOCR(),
            ocr_confidence=.2,
        )
        self.assertGreater(result["report"]["change_count"], 0)
        self.assertEqual(result["before_texts"][0]["text"], "100V")
        self.assertEqual(result["after_texts"][0]["text"], "120V")
        self.assertEqual(len(result["evidence"]["candidates"]),
                         result["report"]["change_count"])
        self.assertEqual(len(result["local_review"]["changes"]),
                         result["report"]["change_count"])
        self.assertIn("graph", result["before_structure"])

    def test_local_review_never_claims_llm_confirmation(self):
        evidence = {
            "candidates": [{
                "change_id": "CHANGE_001",
                "visual": {"added_ink_pixels": 10, "removed_ink_pixels": 1},
                "before": {"ocr": [{"text": "24V"}]},
                "after": {"ocr": [{"text": "48V"}]},
            }]
        }
        review = local_preliminary_review(evidence)
        self.assertEqual(review["analysis_source"], "local_preliminary")
        self.assertEqual(review["changes"][0]["verdict"], "probable")


if __name__ == "__main__":
    unittest.main(verbosity=2)
