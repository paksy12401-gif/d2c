import json
import unittest
from unittest import mock

import llm_revision


def visual_report():
    return {
        "alignment": {"method": "ORB + RANSAC affine", "matched_features": 100,
                      "inlier_ratio": .99},
        "difference_stats": {"changed_ink_pixels": 80},
        "changes": [{"id": "CHANGE_001", "bbox_px": [10, 10, 100, 40],
                     "label": "중앙 · 형상 교체 후보", "source": "raster_difference",
                     "added_ink_pixels": 50, "removed_ink_pixels": 30,
                     "ocr_confidence": None}],
    }


def ocr_report(text):
    return {"runs": {"150": {"image_size": [300, 150], "texts": [
        {"text": text, "confidence": .95, "bbox": [30, 30, 60, 15]}
    ]}}}


def response(value="120 mm"):
    review = {"summary": "치수 변경", "changes": [{
        "change_id": "CHANGE_001", "verdict": "confirmed",
        "change_type": "text_value_changed", "before_value": "100 mm",
        "after_value": value, "affected_object_ids": [],
        "reason": "같은 위치의 치수 문자열이 변경됨", "confidence": .96,
        "needs_review": False,
    }], "global_warnings": []}
    return {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps(review, ensure_ascii=False)}
    ]}]}


class RichRevisionChecks(unittest.TestCase):
    def setUp(self):
        self.payload = llm_revision.build_revision_evidence(
            visual_report(), [200, 100],
            before_ocr_report=ocr_report("100 mm"),
            after_ocr_report=ocr_report("120 mm"),
        )

    def test_rich_payload_contains_actual_revision_evidence_without_images(self):
        candidate = self.payload["candidates"][0]
        self.assertEqual(candidate["before"]["ocr"][0]["text"], "100 mm")
        self.assertEqual(candidate["after"]["ocr"][0]["text"], "120 mm")
        body = llm_revision.request_body(self.payload, "synthetic-model")
        transmitted = body["input"]
        self.assertIn("100 mm", transmitted)
        self.assertIn("120 mm", transmitted)
        self.assertNotIn("image_bytes", transmitted)
        self.assertFalse(body["store"])

    def test_structured_review_is_validated(self):
        connection = mock.MagicMock()
        connection.__enter__.return_value.read.return_value = json.dumps(
            response(), ensure_ascii=False).encode("utf-8")
        opener = mock.Mock()
        opener.open.return_value = connection
        with mock.patch("llm_revision.build_opener", return_value=opener):
            result = llm_revision.request_revision_review(
                self.payload, "synthetic-model", enabled=True,
                api_key="synthetic-key", passes=1,
            )
        self.assertEqual(result["changes"][0]["after_value"], "120 mm")
        request = opener.open.call_args.args[0]
        self.assertIn("100 mm".encode("utf-8"), request.data)

    def test_disabled_or_incomplete_review_fails_closed(self):
        with self.assertRaises(PermissionError):
            llm_revision.request_revision_review(self.payload, "synthetic-model")
        bad = response()
        review = json.loads(bad["output"][0]["content"][0]["text"])
        review["changes"] = []
        bad["output"][0]["content"][0]["text"] = json.dumps(review)
        connection = mock.MagicMock()
        connection.__enter__.return_value.read.return_value = json.dumps(bad).encode()
        opener = mock.Mock()
        opener.open.return_value = connection
        with mock.patch("llm_revision.build_opener", return_value=opener), self.assertRaises(ValueError):
            llm_revision.request_revision_review(
                self.payload, "synthetic-model", enabled=True,
                api_key="synthetic-key", passes=1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
