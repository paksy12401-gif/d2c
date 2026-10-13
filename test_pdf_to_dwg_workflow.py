from pathlib import Path
import unittest

import pymupdf

from pdf_to_dwg_workflow import SAMPLE_PDF, _assign_fresh_entities, convert_pdf, inspect_pdf


class PdfToDwgWorkflowTests(unittest.TestCase):
    def test_vector_pdf_is_accepted_and_image_only_is_rejected(self):
        doc = pymupdf.open()
        page = doc.new_page(width=200, height=100)
        page.draw_line((10, 10), (100, 10))
        data = doc.tobytes()
        doc.close()
        self.assertGreater(inspect_pdf(data)["vector_paths"], 0)

        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        doc = pymupdf.open()
        page = doc.new_page(width=200, height=100)
        page.insert_image(page.rect, pixmap=image)
        with self.assertRaisesRegex(ValueError, "벡터"):
            inspect_pdf(doc.tobytes())
        doc.close()

    def test_block_selection_keeps_through_wire_outside(self):
        def entity(handle, bounds):
            return {"handle": handle, "type": "LWPOLYLINE", "layer": "PDF2_Geometry",
                    "bounds": bounds}

        detection = {"class_name": "7", "status": "strong_candidate",
                     "edge_score": .9, "bbox_pdf_pt": [72, 72, 144, 144]}
        rows = [entity("wire", (.2, 1.5, 3, 1.5)),
                entity("a", (1.2, 1.2, 1.3, 1.3)),
                entity("b", (1.4, 1.2, 1.5, 1.3)),
                entity("c", (1.6, 1.2, 1.7, 1.3))]
        groups = _assign_fresh_entities([detection], rows, 216)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["handles"], ["a", "b", "c"])

    def test_identical_sample_uses_checked_dwg(self):
        if not SAMPLE_PDF.is_file():
            self.skipTest("sample PDF missing")
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            result = convert_pdf(SAMPLE_PDF.read_bytes(), output_root=Path(folder))
            self.assertEqual(result["report"]["mode"], "verified_identical_pdf")
            self.assertEqual(result["report"]["symbol_blocks"], 59)
            self.assertTrue(result["archive"].is_file())


if __name__ == "__main__":
    unittest.main()
