from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pymupdf

from drawing_compare.compare import _save_png
from drawing_comparison_workflow import compare_pdfs, inspect_compare_pdf


def drawing_pdf(*, changed: bool = False) -> bytes:
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=220)
        page.draw_rect(pymupdf.Rect(20, 20, 280, 200), color=(0, 0, 0))
        page.draw_rect(pymupdf.Rect(65, 65, 95, 95), color=(0, 0, 0))
        if changed:
            page.draw_circle(pymupdf.Point(215, 140), 16, color=(0, 0, 0))
        return document.tobytes()


class DrawingComparisonWorkflowTests(unittest.TestCase):
    def test_png_is_saved_in_korean_path(self):
        with tempfile.TemporaryDirectory(prefix="도면_", dir=Path.cwd()) as temporary:
            path = Path(temporary) / "오버레이.png"
            _save_png(path, np.zeros((3, 3, 3), dtype=np.uint8))
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))

    def test_rejects_image_only_pdf(self):
        with pymupdf.open() as document:
            page = document.new_page()
            image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
            page.insert_image(page.rect, pixmap=image)
            with self.assertRaisesRegex(ValueError, "벡터"):
                inspect_compare_pdf(document.tobytes())

    def test_detects_change_and_packages_report(self):
        with tempfile.TemporaryDirectory(prefix="도면_", dir=Path.cwd()) as temporary:
            with patch.dict(os.environ, {
                "DRAWING_COMPARE_LLM_URL": "https://example.invalid/v1/chat/completions",
                "DRAWING_COMPARE_LLM_API_KEY": "test",
                "DRAWING_COMPARE_LLM_MODEL": "test",
            }), patch("urllib.request.urlopen") as remote_call:
                output = compare_pdfs(drawing_pdf(), drawing_pdf(changed=True),
                                      output_root=Path(temporary))
                remote_call.assert_not_called()
            self.assertGreaterEqual(len(output["result"].regions), 1)
            self.assertTrue(output["result"].overlay_path.is_file())
            with zipfile.ZipFile(output["archive"]) as archive:
                self.assertIn("report.html", archive.namelist())
                self.assertIn("result.json", archive.namelist())
                self.assertIn("overlay.png", archive.namelist())
                self.assertTrue(any(name.startswith("regions/") for name in archive.namelist()))

    def test_identical_drawing_has_no_changes(self):
        data = drawing_pdf()
        with tempfile.TemporaryDirectory() as temporary:
            output = compare_pdfs(data, data, output_root=Path(temporary))
            self.assertFalse(output["result"].regions)


if __name__ == "__main__":
    unittest.main()
