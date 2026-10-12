"""Streamlit UI integration using only synthetic input and a mocked API."""
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class AppChecks(unittest.TestCase):
    def test_local_demo_review_and_dxf(self):
        with patch("privacy.request_selection") as api:
            app = AppTest.from_file("app.py", default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            app.button(key="demo").click().run()
            self.assertEqual(len(app.exception), 0)
            app.button(key="confirm").click().run()
            self.assertEqual(len(app.exception), 0)
            geometry = next(c for c in app.checkbox if c.label == "원본과 도형·축척을 로컬에서 비교했습니다")
            geometry.check().run()
            next(b for b in app.button if b.label == "검토용 DXF 생성").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.session_state["dxf"].startswith(b"  0"))
            api.assert_not_called()

    def test_key_reaches_api_only_when_test_button_clicked(self):
        with patch("privacy.request_selection", return_value={"role": "length", "target": "a"}) as api:
            app = AppTest.from_file("app.py", default_timeout=20).run()
            app.text_input(key="api_key").set_value("synthetic-test-key").run()
            api.assert_not_called()
            app.button(key="api_test").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(api.call_args.kwargs["api_key"], "synthetic-test-key")
            self.assertTrue(api.call_args.kwargs["enabled"])


if __name__ == "__main__":
    unittest.main()
