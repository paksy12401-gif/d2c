"""Find short border labels by page geometry; no reference file or sequence repair."""
import json
import re
import sys

import cv2
import numpy as np
import pymupdf as fitz
import torch

from infer_real_validation import render
from iterate_fusion import ROOT, lines
from preserved_reader import PreservedReader

OUT=ROOT.parent/'output/cad_fusion_refined'


def frame_fields(page):
    result=[]
    for index,path in enumerate(page.get_drawings()):
        b=path_bounds(path);horizontal=b.y1<85 or b.y0>page.rect.height-85
        vertical=b.x1<85 or b.x0>page.rect.width-85
        if (horizontal or vertical) and 2<b.height<25 and 0<b.width<50:
            result.append(dict(id=f'F{index}',box_pdf=list(b),cap=b.height,role='numeric' if horizontal else 'letter'))
    return result


def path_bounds(path):
    # Some PDF path rectangles omit an earlier move; endpoints retain the full strokes.
    points=[p for item in path['items'] for p in item[1:] if hasattr(p,'x') and hasattr(p,'y')]
    if not points:return path['rect']
    return fitz.Rect(min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points))


def choose_label(yolo,score,ocr,ocr_score,role):
    # Only the label's role constrains the alphabet; never use expected grid numbers.
    if role=='numeric':
        if re.fullmatch(r'\d{1,3}',ocr) and ocr_score>=.95:return ocr,'ocr_numeric'
        mapped=yolo.replace('O','0')
        if re.fullmatch(r'\d{1,3}',mapped):return mapped,'yolo_numeric'
        return yolo,'unresolved_numeric'
    if re.fullmatch('[A-Z]',yolo) and score>=.85:return yolo,'yolo_letter'
    if re.fullmatch('[A-Z]',ocr) and ocr_score>=.95:return ocr,'ocr_letter'
    return yolo,'unresolved_letter'


def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    page=fitz.open(ROOT/'cad_without_tables/source.pdf')[0];fields=frame_fields(page)
    reader=PreservedReader();jobs=[]
    for field in fields:
        b=field['box_pdf'];cap=field['cap'];scale=56/cap
        image,origin=render(page,[b[0]-cap*.15,b[1]-cap*.15,b[2]+cap*.15,b[3]+cap*.15],scale)
        canvas=np.full((384,384,3),255,np.uint8);h,w=image.shape[:2];x=(384-w)//2;y=(384-h)//2
        if x<0 or y<0:raise ValueError('Unexpected large border label')
        canvas[y:y+h,x:x+w]=image
        crop=cv2.copyMakeBorder(image,8,8,8,8,cv2.BORDER_CONSTANT,value=(255,255,255))
        jobs.append((field,canvas,crop,x,y,origin,scale))
    records=[]
    for start in range(0,len(jobs),16):
        batch=jobs[start:start+16]
        predictions=reader.letters.predict([j[1] for j in batch],imgsz=384,conf=.1,iou=.5,agnostic_nms=True,end2end=False,device=0,batch=16,verbose=False)
        for job,pred in zip(batch,predictions):
            field,_,_,x,y,origin,scale=job;glyphs=[]
            for b in pred.boxes.cpu().numpy().data.tolist():
                if not (x<=(b[0]+b[2])/2<=384-x and y<=(b[1]+b[3])/2<=384-y):continue
                glyphs.append(dict(text=reader.names[int(b[5])],score=b[4],box_pdf=[(b[k]-(x if k%2==0 else y))/scale+origin[k%2] for k in range(4)]))
            text=''.join(''.join(lines(glyphs)).split())
            records.append({**field,'yolo':text,'yolo_score':min((g['score'] for g in glyphs),default=0),'glyphs':glyphs})
    sys.path.insert(0,str(ROOT.parent))
    from models_local import _offline
    from paddle_ocr_local import model_dirs
    with _offline(paddle_cache=ROOT.parent/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        for record,pred in zip(records,model.predict(input=[j[2] for j in jobs],batch_size=8)):
            raw=pred.json.get('res',pred.json);record['ocr']=str(raw.get('rec_text','')).strip();record['ocr_score']=float(raw.get('rec_score',0))
            record['text'],record['selected_engine']=choose_label(record['yolo'],record['yolo_score'],record['ocr'],record['ocr_score'],record['role'])
    result=dict(inference_llm=False,answers_loaded=False,sequence_correction=False,scope='Geometry-based border labels of this PDF; numeric/letter role by border orientation, no expected values used.',records=records)
    (OUT/'frame_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FRAME READING COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
