import unittest
import numpy as np

from improve_title_local import punctuation_kind, owned_window, inject_punctuation, model_supported_punctuation
from retry_table_title_local import crop_image


class TitleTests(unittest.TestCase):
    def test_internal_horizontal_bar_is_not_automatically_punctuation(self):
        group = dict(box=[10, 4, 17, 4], segments=[[10, 4, 17, 4]])
        self.assertIsNone(punctuation_kind(group, 10, 10))

    def test_slash_requires_diagonal_source_segment(self):
        self.assertEqual(punctuation_kind(dict(box=[0, 0, 10, 10],
                         segments=[[0, 10, 10, 0]]), 10, 10), '/')
        self.assertIsNone(punctuation_kind(dict(box=[0, 0, 10, 10],
                          segments=[[0, 0, 10, 10]]), 10, 10))

    def test_parenthesis_like_hangul_part_requires_model_support(self):
        group = dict(box=[0, 0, 2, 10], segments=[[0, 0, 0, 10], [0, 5, 2, 5]])
        self.assertEqual(punctuation_kind(group, 10, 10), ')')
        field = dict(groups=[group], cap=10, baseline=10)
        self.assertEqual(list(model_supported_punctuation(field, [], None)), [])

    def test_owned_crop_is_translation_invariant(self):
        def field(dx, dy):
            return dict(cap=10, groups=[dict(box=[dx, dy, dx+10, dy+10],
                        segments=[[dx, dy, dx+10, dy+10]])])
        a, ab = owned_window(field(0, 0), [0, 0, 10, 10], [])
        b, bb = owned_window(field(500, 300), [500, 300, 510, 310], [])
        ai, ao, _ = crop_image(a, ab, 1, 1)
        bi, bo, _ = crop_image(b, bb, 1, 1)
        np.testing.assert_array_equal(ai, bi)
        np.testing.assert_allclose(np.array(bo)-ao, [500, 300])

    def test_small_punctuation_does_not_remove_neighbor_letter(self):
        glyphs = [dict(text='X', box_pdf=[10, 0, 20, 10]),
                  dict(text='l', box_pdf=[20, 8, 22, 11])]
        punctuation = [dict(text=',', box_pdf=[20, 8, 22, 11])]
        self.assertEqual([g['text'] for g in inject_punctuation(glyphs, punctuation, 10)], ['X', ','])


if __name__ == '__main__':
    unittest.main()
