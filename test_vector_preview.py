import io
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import ezdxf
from PIL import Image

from vision import preprocess
from vectorize_local import extract_vectors
from benchmark_preview import render_dxf
from cad import build_dxf


class VectorChecks(unittest.TestCase):
    def test_preserves_single_pixel_wire(self):
        image=np.full((100,160),255,np.uint8)
        image[50,10:151]=0
        data=io.BytesIO();Image.fromarray(image).save(data,format='PNG')
        gray=preprocess(data.getvalue(),'.png',preserve_thin_lines=True)
        result=extract_vectors(gray)
        lines=[o['geometry'] for o in result['objects'] if o['kind']=='line']
        self.assertTrue(any(abs(g['end'][0]-g['start'][0])>130 for g in lines))

    def test_shallow_rectangle_does_not_become_unsupported_arc(self):
        image=np.full((90,220),255,np.uint8)
        cv2.rectangle(image,(6,7),(183,37),0,1)
        result=extract_vectors(image)
        self.assertTrue(result['objects'])
        self.assertFalse(any(o['kind']=='arc' for o in result['objects']))

    def test_circle_rounding_near_edge_is_bounded(self):
        image=np.full((100,100),255,np.uint8)
        cv2.circle(image,(50,91),8,0,1)
        with patch('vectorize_local.cv2.HoughCircles',return_value=np.array([[[50.,91.5,7.5]]],np.float32)):
            extract_vectors(image)

    def test_polyline_dxf_reflection_and_actual_render_dimensions(self):
        structure={'width':240,'height':180,'objects':[
            {'id':'P','kind':'polyline','geometry':{'points':[[20,20],[100,20],[100,70]],'closed':False}},
            {'id':'C','kind':'circle','geometry':{'center':[170,80],'radius':25}}
        ],'texts':[]}
        data=build_dxf(structure,.1)
        doc=ezdxf.read(io.StringIO(data.decode()))
        vertices=list(doc.modelspace().query('LWPOLYLINE')[0].get_points('xy'))
        np.testing.assert_allclose(vertices,[(2,16),(10,16),(10,11)])
        self.assertFalse(doc.modelspace().query('IMAGE'))
        self.assertEqual(render_dxf(data,240,180).size,(240,180))


if __name__=='__main__':unittest.main()
