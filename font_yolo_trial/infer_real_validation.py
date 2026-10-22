"""Direct YOLO search and fixed-region probes; no answer file, LLM or word correction."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import torch
from ultralytics.engine.results import Results

from preserved_reader import PreservedReader
from prepare import write_image
from train import read_rows
from evaluate_probes import nms

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'real_font_validation'
HEIGHTS=[32,40,56]
SCALES=[3.2,4.0,5.6]
TABLES=[[5045,170,7140,1250],[7990,70,11170,2230]]


def render(page, box, scale):
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),clip=fitz.Rect(box),alpha=False)
    return np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)[:,:,::-1].copy(),[pix.x/scale,pix.y/scale]


def inside(box, target, margin=0):
    x,y=(box[0]+box[2])/2,(box[1]+box[3])/2
    return target[0]-margin<=x<=target[2]+margin and target[1]-margin<=y<=target[3]+margin


def pdf_boxes(result, origin, scale):
    return [[b[0]/scale+origin[0],b[1]/scale+origin[1],b[2]/scale+origin[0],b[3]/scale+origin[1],*b[4:]]
            for b in result.boxes.cpu().numpy().data.tolist()]


def as_lines(boxes, names):
    result=Results(orig_img=np.full((1,1,3),255,dtype=np.uint8),path='coordinates_only',names=names,
                   boxes=np.asarray(boxes,dtype=np.float32).reshape(-1,6))
    return read_rows(result)


def probes(page, manifest, reader):
    records=[];korean_jobs=[]
    for region in manifest['regions']:
        box=region['box_pdf'];x0,y0,x1,y1=box;cap=region['reference_cap_height_pdf']
        for view in ['isolated','context']:
            px,py=(.2,.2) if view=='isolated' else (3,3)
            clip=[max(0,x0-cap*px),max(0,y0-cap*py),min(page.rect.width,x1+cap*px),min(page.rect.height,y1+cap*py)]
            for height in HEIGHTS:
                scale=height/cap;image,origin=render(page,clip,scale)
                name=f"{region['id']}_{view}_{height}"
                if region['kind']=='korean':
                    # Recognition receives the complete line, not surrounding neighboring captions.
                    crop,_=render(page,[x0-cap*.15,y0-cap*.2,x1+cap*.15,y1+cap*.2],scale)
                    korean_jobs.append((dict(id=region['id'],view=view,target_height=height,engine='korean_ocr',
                                              note='Same direct line crop for both views; context experiment is YOLO-only.'),crop))
                    continue
                result,baseline,details=reader.predict(image,mode='tiles')
                boxes=[b for b in pdf_boxes(result,origin,scale) if inside(b,box,cap*.035)]
                base=[b for b in pdf_boxes(baseline,origin,scale) if inside(b,box,cap*.035)]
                entry=dict(id=region['id'],view=view,target_height=height,engine='yolo',scale=scale,
                           clip_pdf=clip,raw_lines=as_lines(boxes,reader.names),baseline_lines=as_lines(base,reader.names),
                           boxes=boxes,all_box_count=len(result.boxes),selected_box_count=len(boxes))
                records.append(entry)
                selected_pixels=np.asarray([[*(np.array(b[:4])-np.tile(origin,2))*scale,*b[4:]] for b in boxes],dtype=np.float32).reshape(-1,6)
                panel=Results(orig_img=image,path=name,names=reader.names,boxes=selected_pixels)
                write_image(OUT/f'{name}_detected.png',panel.plot(line_width=1,font_size=9))
        print(f"PROBE {region['id']} done",flush=True)
        (OUT/'probe_progress.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    if korean_jobs:
        sys.path.insert(0,str(ROOT.parent))
        from models_local import _offline
        from paddle_ocr_local import model_dirs
        with _offline(paddle_cache=ROOT.parent/'.model_cache/paddlex'):
            from paddleocr import TextRecognition
            _,rec=model_dirs()
            model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(rec),device='cpu',
                                  enable_mkldnn=False,cpu_threads=4)
            results=list(model.predict(input=[im for _,im in korean_jobs],batch_size=4))
            assert len(results)==len(korean_jobs)
            for (entry,_),result in zip(korean_jobs,results):
                raw=result.json.get('res',result.json)
                records.append({**entry,'raw_lines':[str(raw.get('rec_text',''))],'score':float(raw.get('rec_score',0))})
    return records


def scan(page, reader):
    # These are sheet-content rectangles, independent of text proposals or validation labels.
    areas=[[75,75,5250,2250],[6000,1270,7160,1610]]
    records=[]
    for scale in SCALES:
        boxes=[]
        for area_index,area in enumerate(areas):
            ys=list(np.arange(area[1],area[3],288/scale))
            for i,y in enumerate(ys):
                clip=[area[0],y,area[2],min(area[3],y+384/scale)]
                image,origin=render(page,clip,scale)
                for t in TABLES:
                    a,b,c,d=[round((t[j]-origin[j%2])*scale) for j in range(4)]
                    if c>0 and d>0 and a<image.shape[1] and b<image.shape[0]:
                        image[max(0,b):max(0,min(d,image.shape[0])),max(0,a):max(0,min(c,image.shape[1]))]=255
                result,_,_=reader.predict(image,mode='tiles')
                lower=y+(48/scale if i else 0)
                upper=y+(336/scale if y+384/scale<area[3] else 384/scale)
                for box in pdf_boxes(result,origin,scale):
                    cy=(box[1]+box[3])/2
                    if lower<=cy<upper and not any(inside(box,t) for t in TABLES):boxes.append(box)
                print(f'SCAN scale={scale} area={area_index+1} band={i+1}/{len(ys)} boxes={len(boxes)}',flush=True)
                (OUT/'scan_progress.json').write_text(json.dumps({'scale':scale,'boxes':boxes}),encoding='utf-8')
        merged=nms(boxes,.5)
        records.append(dict(scale=scale,boxes=merged,pre_nms_count=len(boxes),post_nms_count=len(merged)))
        (OUT/'scan_checkpoint.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    return dict(areas_pdf=areas,records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['probes','scan'],required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    manifest=json.loads((OUT/'regions.json').read_text(encoding='utf-8'))
    source=Path(manifest['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
    reader=PreservedReader();page=fitz.open(source)[0]
    payload={'llm_inference':False,'word_correction':False,'training':False,'paddle_proposal_gate':False,
             'source_sha256':manifest['source_sha256'],'regions_sha256':hashlib.sha256((OUT/'regions.json').read_bytes()).hexdigest(),
             'model_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in reader.paths.items()},'names':reader.names,
             'confidence':.1,'tile_size':384,'tile_stride':288}
    if args.stage=='probes':payload['records']=probes(page,manifest,reader)
    else:payload.update(scan(page,reader))
    (OUT/f'{args.stage}_predictions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{args.stage.upper()} COMPLETE',flush=True)


if __name__=='__main__':main()
