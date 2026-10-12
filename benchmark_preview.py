"""Reproducible local comparison: legacy Hough vs OCR + centerline vectors.

Pixel fidelity is NOT symbol accuracy, electrical connectivity or CAD correctness.
Both methods use the same input scale and the same exclusion mask for OCR regions.
No ground-truth CAD is available for the supplied samples.
"""
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ezdxf

from cad import build_dxf
from vision import preprocess, extract_geometry
from vectorize_local import extract_vectors, ink_mask, text_mask

ROOT = Path(__file__).parent


def dxf_geometry_mask(data, width, height, scale=.1):
    """Rasterize parsed DXF geometry at original pixels, without TEXT, for scoring."""
    doc=ezdxf.read(StringIO(data.decode('utf-8')))
    canvas=np.zeros((height,width),np.uint8)
    def xy(p):return (round(p[0]/scale),round(height-p[1]/scale))
    for e in doc.modelspace():
        kind=e.dxftype()
        if kind=='LINE':cv2.line(canvas,xy(e.dxf.start),xy(e.dxf.end),255,1)
        elif kind=='CIRCLE':cv2.circle(canvas,xy(e.dxf.center),max(1,round(e.dxf.radius/scale)),255,1)
        elif kind=='LWPOLYLINE':
            points=np.array([xy(p) for p in e.get_points('xy')],np.int32)
            cv2.polylines(canvas,[points],e.closed,255,1)
        elif kind=='ARC':
            start=e.dxf.start_angle; span=(e.dxf.end_angle-start)%360
            angles=np.radians(np.linspace(start,start+span,max(8,int(span*e.dxf.radius/scale/40))))
            points=np.array([xy((e.dxf.center.x+e.dxf.radius*np.cos(a),e.dxf.center.y+e.dxf.radius*np.sin(a))) for a in angles],np.int32)
            cv2.polylines(canvas,[points],False,255,1)
    return canvas>0


def pixel_metrics(gray, geometry, texts, tolerance=2):
    from skimage.morphology import skeletonize
    keep=~text_mask(gray.shape,texts)
    reference=skeletonize(ink_mask(gray) & keep)
    predicted=geometry & keep
    near_ref=cv2.distanceTransform((~reference).astype(np.uint8),cv2.DIST_L2,3)
    near_pred=cv2.distanceTransform((~predicted).astype(np.uint8),cv2.DIST_L2,3)
    precision=float((near_ref[predicted]<=tolerance).mean()) if predicted.any() else 0.
    recall=float((near_pred[reference]<=tolerance).mean()) if reference.any() else 0.
    return {'geometry_ink_precision':precision,'geometry_ink_recall':recall,
            'geometry_ink_f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'tolerance_px':tolerance,'reference_skeleton_pixels':int(reference.sum()),
            'predicted_pixels':int(predicted.sum()),
            'definition':'Raster ink agreement outside detected OCR boxes. Not semantic/electrical accuracy.'}


def render_dxf(data, width, height, scale=.1):
    """Render the actual exported DXF, including editable TEXT, not the input raster."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from ezdxf.addons.drawing.config import Configuration,ColorPolicy,BackgroundPolicy
    doc=ezdxf.read(StringIO(data.decode('utf-8')))
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.is_file():
        for style in doc.styles:style.dxf.font=str(font)
    fig=Figure(figsize=(width/100,height/100),dpi=100,facecolor='white')
    canvas=FigureCanvasAgg(fig)
    ax=fig.add_axes([0,0,1,1])
    config=Configuration(color_policy=ColorPolicy.BLACK,background_policy=BackgroundPolicy.WHITE,
                         min_lineweight=.06)
    Frontend(RenderContext(doc),MatplotlibBackend(ax,adjust_figure=False),config=config).draw_layout(doc.modelspace(),finalize=True)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(0,width*scale);ax.set_ylim(0,height*scale);ax.set_axis_off()
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert('RGB')


def compare_file(path, output_dir, *, ocr_engine="paddle", reader=None):
    path=Path(path);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    raw=path.read_bytes()
    gray=preprocess(raw,path.suffix,preserve_thin_lines=True)
    started=time.perf_counter()
    if ocr_engine == "paddle":
        from paddle_ocr_local import read_text
        texts=read_text(gray,reader,min_confidence=.5)
    elif ocr_engine == "easy":
        from ocr_local import read_text
        texts=read_text(gray,reader)
    elif ocr_engine == "none":texts=[]
    else:raise ValueError("ocr_engine은 paddle/easy/none 중 하나입니다.")
    ocr_seconds=time.perf_counter()-started
    start=time.perf_counter()
    baseline=extract_geometry(preprocess(raw,path.suffix))
    legacy_seconds=time.perf_counter()-start
    start=time.perf_counter()
    improved=extract_vectors(gray,texts)
    vector_seconds=time.perf_counter()-start
    reports={}
    images={'original':Image.fromarray(gray).convert('RGB')}
    for name,structure,seconds in [('baseline',baseline,legacy_seconds),('improved',improved,vector_seconds)]:
        data=build_dxf(structure,.1)
        (out/f'{name}.dxf').write_bytes(data)
        (out/f'{name}.json').write_text(json.dumps(structure,ensure_ascii=False,indent=2),encoding='utf-8')
        mask=dxf_geometry_mask(data,structure['width'],structure['height'])
        images[name]=render_dxf(data,structure['width'],structure['height'])
        reports[name]={'vectorization_seconds':round(seconds,3),
                       'entities':dict(Counter(o['kind'] for o in structure['objects'])),
                       'ocr_texts':len(structure['texts']),**pixel_metrics(gray,mask,texts)}
    for name,img in images.items():img.save(out/f'{name}.png')
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
    w,h=images['original'].size
    comparison=Image.new('RGB',(w*3,h+65),'#eef2f6')
    draw=ImageDraw.Draw(comparison)
    for i,(name,title) in enumerate([('original','원본 도면'),('baseline','기존: Hough 직선'),('improved','개선: 로컬 OCR + 중심선·원·곡선')]):
        comparison.paste(images[name],(i*w,65));draw.text((i*w+12,15),title,font=font,fill='#172336')
    comparison.save(out/'comparison.png')
    report={'input_sha256':hashlib.sha256(raw).hexdigest(),'input_name':path.name,
            'image_size':[w,h],'ocr_seconds_including_load_if_uncached':round(ocr_seconds,3),
            'api_calls':0,'ocr_engine':ocr_engine,'scale_mm_per_pixel':.1,'scale_calibrated':False,
            'ground_truth':'No annotated CAD or independent electrical test labels. Single-sample visual fidelity only.',
            'methods':reports,'texts':texts}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    import argparse,os,faulthandler
    faulthandler.enable()
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(2)
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output',default='output/reference_comparison');p.add_argument('--ocr-engine',choices=['paddle','easy','none'],default='paddle')
    args=p.parse_args()
    print(json.dumps(compare_file(args.image,args.output,ocr_engine=args.ocr_engine),ensure_ascii=False,indent=2))
