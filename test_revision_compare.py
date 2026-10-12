"""Synthetic tests for local two-drawing comparison; no model or network."""
import unittest

import cv2
import numpy as np

from revision_compare import compare_documents, serializable_report


def drawing_bytes(*, red_note=False, extra_wire=False):
    image = np.full((600, 900, 3), 255, np.uint8)
    for x in range(40, 861, 80):
        cv2.line(image, (x, 40), (x, 560), (0, 0, 0), 1)
    for y in range(40, 561, 65):
        cv2.line(image, (40, y), (860, y), (0, 0, 0), 1)
    for index, x in enumerate(range(80, 801, 120)):
        cv2.circle(image, (x, 180 + (index % 3) * 100), 22, (0, 0, 0), 2)
        cv2.putText(image, f"C{index:02d}", (x - 25, 175 + (index % 3) * 100),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
    if extra_wire:
        cv2.line(image, (200, 510), (700, 510), (0, 0, 0), 4)
    if red_note:
        # OpenCV uses BGR; pure red becomes RGB red after encoding/decoding.
        cv2.rectangle(image, (180, 470), (720, 540), (0, 0, 255), 3)
        cv2.putText(image, "WIRE CHANGED", (250, 465), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (0, 0, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class RevisionCompareChecks(unittest.TestCase):
    def test_red_revision_is_aligned_listed_and_serializable(self):
        result = compare_documents(
            drawing_bytes(), drawing_bytes(red_note=True, extra_wire=True),
            "png", "png", dpi=100, ocr_reader=None,
        )
        report = serializable_report(result)
        self.assertGreater(report["alignment"]["matched_features"], 20)
        self.assertGreater(report["alignment"]["inlier_ratio"], 0.8)
        self.assertGreater(report["red_annotation_pixels"], 100)
        self.assertGreaterEqual(report["change_count"], 1)
        self.assertTrue(all("before_crop" not in item for item in report["changes"]))
        self.assertTrue(all(item["before_crop"].size and item["after_crop"].size
                            for item in result["items"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
