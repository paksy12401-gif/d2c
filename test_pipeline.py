"""Run: python -m unittest -v test_pipeline.py

Synthetic data only. Every HTTP opener is mocked; no API/CAD process is started.
"""

import copy
import io
import json
import unittest
from unittest import mock

import ezdxf

import cad
import pipeline
import privacy


def synthetic_structure():
    return {
        "width": 1000, "height": 800,
        "objects": [{"id": "WIRE_LOCAL_ONLY", "kind": "wire",
                     "geometry": {"start": [30, 50], "end": [830, 50]}}],
        "texts": [{"id": "TEXT_LOCAL_ONLY", "text": "150", "bbox": [100, 20, 30, 12],
                   "links": [{"object_id": "WIRE_LOCAL_ONLY", "evidence": "dimension_line"}]}],
    }


def api_response(selection=None):
    return {"status": "completed", "output": [{"type": "message", "status": "completed", "content": [
        {"type": "output_text", "text": json.dumps(selection or {"role": "length", "target": "a"})}
    ]}]}


class BoundaryChecks(unittest.TestCase):
    def setUp(self):
        self.structure = synthetic_structure()
        self.row = pipeline.analyze(self.structure)[0]
        self.payload, self.mapping = privacy.make_payload(self.row)

    def call_mock_api(self, response, payload=None):
        raw = response if isinstance(response, bytes) else json.dumps(response).encode("utf-8")
        connection = mock.MagicMock()
        connection.__enter__.return_value.read.return_value = raw
        opener = mock.Mock()
        opener.open.return_value = connection
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "synthetic-test-key"}), \
                mock.patch("privacy.build_opener", return_value=opener):
            selected = privacy.request_selection(payload or self.payload, "synthetic-model", enabled=True)
        request = opener.open.call_args.args[0]
        return selected, request

    def test_actual_http_bytes_contain_only_allowlisted_categories(self):
        secrets = ["PRIVATE_RAW_TEXT_7921", "PRIVATE_OBJECT_ID_8823", "PRIVATE_FILENAME_1847"]
        row = {**self.row, "raw_text": secrets[0], "value": 984731.625,
               "filename": secrets[2], "bbox": [9123, 8123, 17, 11],
               "full_structure": self.structure,
               "candidates": [{"object_id": secrets[1], "kind": "wire",
                               "evidence": "dimension_line", "roles": ["length"]}]}
        payload, mapping = privacy.make_payload(row)
        selection, request = self.call_mock_api(api_response(), payload)
        self.assertEqual(selection, {"role": "length", "target": "a"})
        self.assertEqual(mapping, {"a": secrets[1]})
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer synthetic-test-key")
        body = json.loads(request.data.decode("utf-8"))
        self.assertFalse(body["store"])
        self.assertNotIn("synthetic-test-key", request.data.decode("utf-8"))
        transmitted = json.loads(body["input"])
        self.assertEqual(transmitted, {
            "token_kind": "bare_number", "unit": "unknown",
            "options": [{"target": "a", "kind": "wire", "evidence": "dimension_line", "roles": ["length"]}],
        })
        for secret in secrets + ["984731.625", "9123", "8123", "WIRE_LOCAL_ONLY", "TEXT_LOCAL_ONLY"]:
            self.assertNotIn(secret.encode("utf-8"), request.data)
        schema = body["text"]["format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"role", "target"})
        self.assertFalse(schema["additionalProperties"])

    def test_payload_rejects_raw_fields_unknown_values_and_bad_types_before_http(self):
        bad_payloads = []
        for key in ("raw_text", "image", "pdf", "value", "bbox", "objects", "filename"):
            bad_payloads.append({**self.payload, key: "SYNTHETIC_SECRET"})
            nested = copy.deepcopy(self.payload)
            nested["options"][0][key] = "SYNTHETIC_SECRET"
            bad_payloads.append(nested)
        for key, value in [("token_kind", "identifier"), ("token_kind", []), ("unit", []), ("options", [])]:
            bad_payloads.append({**self.payload, key: value})
        for key, value in [("target", "real-part-number"), ("kind", []), ("evidence", []),
                           ("roles", ["length", "length"]), ("roles", ["unknown_role"])]:
            nested = copy.deepcopy(self.payload)
            nested["options"][0][key] = value
            bad_payloads.append(nested)
        with mock.patch("privacy.build_opener") as opener:
            for bad in bad_payloads:
                with self.subTest(payload=bad), self.assertRaises(ValueError):
                    privacy.request_selection(bad, "synthetic-model", enabled=True)
            opener.assert_not_called()

    def test_api_disabled_never_constructs_connection(self):
        with mock.patch("privacy.build_opener") as opener:
            for enabled in (False, None, "true", 1):
                with self.subTest(enabled=enabled), self.assertRaises(PermissionError):
                    privacy.request_selection(self.payload, "synthetic-model", enabled=enabled)
            opener.assert_not_called()
        with self.assertRaises(ValueError):
            privacy.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://invalid.example")

    def test_response_rejects_unknown_target_role_extra_fields_refusal_and_malformed(self):
        responses = [
            api_response({"role": "length", "target": "unknown-target"}),
            api_response({"role": "voltage", "target": "a"}),
            api_response({"role": "length", "target": "a", "value": 999}),
            api_response({"role": "unknown", "target": "a"}),
            {"status": "incomplete", "output": []},
            {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]},
            {"status": "completed", "output": None},
            {"status": "completed", "output": ["bad-message"]},
            {"status": "completed", "output": [{"type": "message", "content": None}]},
            {"status": "completed", "output": [{"type": "message", "content": [None]}]},
            {"status": "completed", "output": [{"type": "message", "status": "incomplete", "content": []}]},
            {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "not-json"}]}]},
            {"status": "completed", "output": []},
            b"not-json", b"\xff", b" " * 65537,
        ]
        for response in responses:
            with self.subTest(response=str(response)[:120]), self.assertRaises(ValueError):
                self.call_mock_api(response)
        selected, _ = self.call_mock_api(api_response({"role": "unknown", "target": None}))
        self.assertEqual(selected, {"role": "unknown", "target": None})

    def test_original_value_and_geometry_are_combined_only_locally(self):
        original = copy.deepcopy(self.structure)
        selected, _ = self.call_mock_api(api_response())
        unresolved = pipeline.finalize(self.structure, {})
        self.assertEqual(unresolved["objects"][0]["attributes"], {})
        self.assertEqual(unresolved["status"], "needs_review")
        # This is an explicit local review decision, not an automatic API commit.
        reviewed = {self.row["text_id"]: {"role": selected["role"], "object_id": self.mapping[selected["target"]]}}
        final = pipeline.finalize(self.structure, reviewed)
        attribute = final["objects"][0]["attributes"]["length"]
        self.assertEqual(attribute["value"], 150)
        self.assertEqual(attribute["source_text_id"], "TEXT_LOCAL_ONLY")
        self.assertEqual(attribute["verification"], "user_confirmed")
        self.assertEqual(final["objects"][0]["geometry_px"], original["objects"][0]["geometry"])
        self.assertEqual(self.structure, original)
        drawing = ezdxf.read(io.StringIO(cad.build_dxf(self.structure, 0.25, semantics=final).decode("utf-8")))
        line = drawing.modelspace().query("LINE")[0]
        self.assertAlmostEqual((line.dxf.end - line.dxf.start).magnitude, 200)
        metadata = json.loads("".join(tag.value for tag in line.get_xdata(cad.XDATA_APP)))
        self.assertEqual(metadata["object_id"], "WIRE_LOCAL_ONLY")
        self.assertEqual(metadata["attributes"]["length"]["value"], 150)

    def test_local_rule_types_and_conflicts(self):
        rules = pipeline.load_rules()
        for text, role, value in [("Φ12", "diameter", 12), ("24V", "voltage", 24),
                                  ("0.5SQ", "wire_size", 0.5), ("150 mm", "length", 150)]:
            classified = pipeline.classify(text, rules)
            self.assertEqual(classified["roles"], [role])
            self.assertEqual(classified["value"], value)
        duplicate = copy.deepcopy(self.structure)
        duplicate["texts"].append({**duplicate["texts"][0], "id": "SECOND_TEXT", "text": "170"})
        with self.assertRaises(ValueError):
            pipeline.finalize(duplicate, {key: {"role": "length", "object_id": "WIRE_LOCAL_ONLY"}
                                          for key in ("TEXT_LOCAL_ONLY", "SECOND_TEXT")})

    def test_cad_synthetic_roundtrip(self):
        cad.self_check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
