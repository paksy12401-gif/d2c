"""Local YOLO/PaddleOCR experiment. No LLM or dictionary correction."""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models_local import _offline
from paddle_ocr_local import model_dirs
from mixed_script_reader import glyph_rows, recognition_regions, korean_body, merge_korean
from preserved_reader import PreservedReader
from prepare import write_image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'font_yolo_trial/cad_without_tables'
TABLES = [[5045, 170, 7140, 1250], [7990, 70, 11170, 2230]]


def intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def render(page, box, scale):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=fitz.Rect(box), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy(), (pix.x/scale, pix.y/scale)


def main():
    torch.set_num_threads(4)
    doc = fitz.open(OUT / 'source.pdf'); page = doc[0]
    reader = PreservedReader()
    det, rec = model_dirs()
    proposals = []
    with _offline(paddle_cache=ROOT / '.model_cache/paddlex'):
        from paddleocr import PaddleOCR, TextRecognition
        pipeline = PaddleOCR(text_detection_model_name='PP-OCRv5_server_det', text_detection_model_dir=str(det),
            text_recognition_model_name='korean_PP-OCRv5_mobile_rec', text_recognition_model_dir=str(rec),
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            device='cpu', enable_mkldnn=False, cpu_threads=4,
            text_det_limit_side_len=960, text_det_limit_type='max')
        # Cover all non-table areas, including the separate work notes.
        areas = [[0, 0, 5045, page.rect.height], [5045, 1250, 7990, page.rect.height],
                 [5045, 0, 7990, 170], [7140, 170, 7990, 1250],
                 [11170, 0, page.rect.width, page.rect.height], [7990, 2230, 11170, page.rect.height]]
        tiles = []
        for area in areas:
            for y in np.arange(area[1], area[3], 520):
                for x in np.arange(area[0], area[2], 900):
                    tiles.append([x, y, min(x+1000, area[2]), min(y+620, area[3])])
        for i, tile in enumerate(tiles):
            image, origin = render(page, tile, 2)
            if np.mean(image < 220) < .0002:
                continue
            pred = list(pipeline.predict(input=image, text_rec_score_thresh=0.0))[0]
            raw = pred.json.get('res', pred.json)
            for poly, text, score in zip(raw['rec_polys'], raw['rec_texts'], raw['rec_scores']):
                points = np.asarray(poly)/2 + origin
                box = [*points.min(0), *points.max(0)]
                if any(intersects(box, t) for t in TABLES):
                    continue
                if any(abs((box[0]+box[2]-p['box'][0]-p['box'][2])/2) < 3 and
                       abs((box[1]+box[3]-p['box'][1]-p['box'][3])/2) < 3 for p in proposals):
                    continue
                proposals.append({'box': box, 'paddle_raw': str(text), 'paddle_score': float(score)})
            print(f'DETECT_TILE {i+1}/{len(tiles)} proposals={len(proposals)}', flush=True)
            (OUT/'text_proposals.json').write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding='utf-8')
        del pipeline
        (OUT/'text_proposals.json').write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding='utf-8')
        recognizer = TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(rec), device='cpu', enable_mkldnn=False,cpu_threads=4)
        results = []
        for i, proposal in enumerate(proposals):
            x0,y0,x1,y1 = proposal['box']; cap=y1-y0
            # Include whole outlined characters when the detector found only their upper strokes.
            clip=[max(0,x0-cap*.3), max(0,y0-cap*.6), min(page.rect.width,x1+cap*.3), min(page.rect.height,y1+cap*.6)]
            scale = min(8, max(2, 40/max(cap, 1)))
            image, origin = render(page, clip, scale)
            glyphs, _, details = reader.predict(image, mode='tiles')
            boxes = glyphs.boxes.cpu().numpy().data.tolist()
            polygon = ((np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])-origin)*scale).tolist()
            regions = recognition_regions([polygon], glyph_rows(boxes), details['cleaned'])
            for region in regions:
                a,b,c,d=region['xyxy']
                crop=cv2.copyMakeBorder(details['cleaned'][b:d,a:c], 12,12,12,12,cv2.BORDER_CONSTANT,value=(255,255,255))
                raw=list(recognizer.predict(input=crop, batch_size=1))[0].json
                raw=raw.get('res',raw)
                region.update(text=str(raw.get('rec_text','')),score=float(raw.get('rec_score',0)))
                region['korean_text']=korean_body(region['text'],region['score'])
            merged,names,_=merge_korean(boxes,reader.names,regions)
            candidates=[]
            for box in merged.tolist():
                label=names[int(box[5])]
                pb=[box[0]/scale+origin[0],box[1]/scale+origin[1],box[2]/scale+origin[0],box[3]/scale+origin[1]]
                cx,cy=(pb[0]+pb[2])/2,(pb[1]+pb[3])/2
                if not (x0-cap*.15<=cx<=x1+cap*.15 and y0-cap*.6<=cy<=y1+cap*.6):
                    continue
                if any(intersects(pb,t) for t in TABLES):
                    continue
                candidates.append({'text':label,'score':box[4],'box_pdf':pb,'kind':'korean' if int(box[5])>=67 else 'yolo', 'proposal':i})
            results.extend(candidates)
            if i < 8:
                write_image(OUT/f'proposal_{i:03d}.png', image)
            print(f'READ_REGION {i+1}/{len(proposals)} glyphs={len(candidates)}',flush=True)
            (OUT/'ocr_progress.json').write_text(json.dumps(results,ensure_ascii=False),encoding='utf-8')
    # Spatial duplicate suppression does not change a predicted character.
    kept=[]
    for item in sorted(results,key=lambda r:-r['score']):
        a=item['box_pdf']
        duplicate=False
        for old in kept:
            b=old['box_pdf']; intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            smaller=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
            if intersection/max(smaller,1e-9)>.65:
                duplicate=True;break
        if not duplicate:kept.append(item)
    report={'llm_used':False,'word_correction':False,'training':False,'table_exclusions_pdf':TABLES,
            'source_sha256':hashlib.sha256((OUT/'source.pdf').read_bytes()).hexdigest(),
            'yolo_threshold':.1,'detections':kept,'proposals':proposals,
            'models':{k:str(v) for k,v in reader.paths.items()},
            'model_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in reader.paths.items()},
            'korean_model':str(rec)}
    (OUT/'ocr_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'OCR_COMPLETE detections={len(kept)}',flush=True)


if __name__=='__main__':
    assert intersects([0,0,2,2],[1,1,3,3]) and not intersects([0,0,1,1],[1,1,2,2])
    main()
