import unittest
import ezdxf
from ezdxf.enums import TextEntityAlignment

from auto_large_text_pdf import large_text_fields, existing_field_text


class LargeTextTests(unittest.TestCase):
    def candidates(self, scale=1, dx=0):
        return [dict(path_id=i, cap=h*scale, box=[dx, 0, dx+10*scale, h*scale])
                for i, h in enumerate([10]*20+[11]*5+[27]*2+[63])]

    def test_size_route_is_independent_of_position_and_uniform_scale(self):
        first, sizing = large_text_fields(self.candidates())
        other, _ = large_text_fields(self.candidates(7, 500))
        self.assertEqual([f['path_id'] for f in first], [f['path_id'] for f in other])
        self.assertEqual(len(first), 3)
        self.assertEqual(sizing['body_height'], 10)
        self.assertEqual([f['relative_height'] for f in first], [f['relative_height'] for f in other])

    def test_empty_and_uniform_text_do_not_invent_large_fields(self):
        self.assertEqual(large_text_fields([])[0], [])
        self.assertEqual(large_text_fields(self.candidates()[:20])[0], [])

    def test_existing_id_at_wrong_location_does_not_suppress_detection(self):
        doc = ezdxf.new(); doc.appids.new('CAD_OCR_REVIEW')
        model = doc.modelspace(); e = model.add_text('X', dxfattribs={'height':10/72})
        e.set_placement((20/72, 80/72), (60/72, 80/72), align=TextEntityAlignment.FIT)
        e.set_xdata('CAD_OCR_REVIEW', [(1000, 'P1')])
        field = dict(id='P1', cap=10, box=[20, 10, 60, 20], baseline=20)
        self.assertEqual(existing_field_text(field, model, 100), e.dxf.handle)
        field['box'] = [120, 10, 160, 20]
        self.assertIsNone(existing_field_text(field, model, 100))


if __name__ == '__main__':
    unittest.main()
