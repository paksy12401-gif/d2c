"""Synthetic validation; no real symbol labels, model training or CAD execution."""
import json
from pathlib import Path
import tempfile
import unittest

import pypdfium2 as pdfium
from PIL import Image
from roi_tools import render_roi, save_roi, export_dataset, nms, make_block_lisp

def temporary_directory():
    root = Path(__file__).resolve().parent / "tmp_tests"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class RoiWorkflowChecks(unittest.TestCase):
    def test_render_exact_size_and_page_edge_padding(self):
        with temporary_directory() as folder:
            pdf_path = Path(folder) / "blank.pdf"
            with pdfium.PdfDocument.new() as document:
                document.new_page(720, 360)
                document.save(pdf_path)
            image, meta = render_roi(pdf_path, 1, [1100, 500], 120)
            self.assertEqual(image.size, (640, 640))
            self.assertEqual(meta["valid_size_px"], [100, 100])
            self.assertEqual(image.getpixel((639, 639)), (255, 255, 255))

    def test_label_normalization_and_separated_split(self):
        with temporary_directory() as folder:
            root = Path(folder)
            for x, color in [(0, "white"), (900, "gray")]:
                image = Image.new("RGB", (640, 640), color)
                meta = {"source_pdf": "test.pdf", "page": 1, "origin_px": [x, 0],
                        "dpi": 150, "valid_size_px": [640, 640], "size_px": [640, 640]}
                save_roi(root, image, meta, [{"class_id": 0, "xyxy": [64, 128, 192, 256]}],
                         ["A"], reviewed=True)
            dataset, stats = export_dataset(root)
            self.assertEqual(stats["train"]["images"], 1)
            self.assertEqual(stats["val"]["images"], 1)
            label = next((dataset.parent / "labels" / "train").glob("*.txt"))
            self.assertEqual(label.read_text(), "0 0.20000000 0.30000000 0.20000000 0.20000000")

    def test_overlapping_crops_are_not_split_across_train_val(self):
        with temporary_directory() as folder:
            root = Path(folder)
            for x, color in [(0, "white"), (50, "gray")]:
                meta = {"source_pdf": "test.pdf", "page": 1, "origin_px": [x, 0],
                        "dpi": 150, "valid_size_px": [640, 640]}
                save_roi(root, Image.new("RGB", (640, 640), color), meta,
                         [{"class_id": 0, "xyxy": [10, 10, 20, 20]}], ["A"], reviewed=True)
            with self.assertRaisesRegex(ValueError, "All ROIs overlap"):
                export_dataset(root)

    def test_nms_is_class_and_page_aware(self):
        base = {"page": 1, "class_id": 0, "confidence": .9, "bbox_px": [10, 10, 40, 40]}
        result = nms([base, {**base, "confidence": .8},
                      {**base, "class_id": 1}, {**base, "page": 2}])
        self.assertEqual(len(result), 3)

    def test_block_export_requires_review_and_mapping(self):
        detection = {"id": "SYMBOL_0001", "class_name": "A", "class_id": 0, "bbox_pt_top": [10, 20, 40, 50],
                     "approved": False, "rotation_deg": 0}
        report = {"page_size_pt": [100, 100], "detections": [detection]}
        with temporary_directory() as folder:
            output = Path(folder) / "test.lsp"
            with self.assertRaises(ValueError):
                make_block_lisp(report, {}, output)
            detection["approved"] = True
            mapping = {"A": {"block_name": "A_BLOCK", "import_layer": "PDF_Geometry",
                             "reference_width": 5.0, "anchor": [.5, .5]}}
            make_block_lisp(report, mapping, output, mode="replace_library")
            source = output.read_text(encoding="ascii")
            self.assertIn('"_W"', source)
            self.assertIn('"Yes"', source)
            self.assertIn("vl-catch-all-error-p", source)
            # Check delimiter balance outside LISP strings and comments.
            balance, quoted, escape = 0, False, False
            for line in source.splitlines():
                for c in line:
                    if not quoted and c == ";":
                        break
                    if c == '"' and not escape:
                        quoted = not quoted
                    if not quoted:
                        balance += (c == "(") - (c == ")")
                        self.assertGreaterEqual(balance, 0)
                    escape = c == "\\" and not escape
                escape = False
            self.assertEqual(balance, 0)
            self.assertFalse(quoted)
            make_block_lisp(report, {}, output, mode="group_imported")
            source = output.read_text(encoding="ascii")
            self.assertIn('(sr:group "SYMBOL_0001"', source)
            self.assertIn("vla-CopyObjects", source)


if __name__ == "__main__":
    unittest.main()
