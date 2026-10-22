"""Freeze stroke-part diagnostics for every saved alphanumeric proposal; no answers."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from infer_real_validation import render
from stroke_template_reader import StrokeReader,exemplar_images,calibrate

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'output/cad_stroke_templates'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    exemplars=exemplar_images(ROOT/'punctuation_fixed/glyphs')
    reader=StrokeReader(exemplars)
    limits,calibration=calibrate(reader,exemplars)
    (OUT/'calibration.json').write_text(json.dumps(dict(scope='Same exemplars, raster size and thickness changes; not independent font validation.',limits=limits,rows=calibration),ensure_ascii=False,indent=2),encoding='utf-8')
    print('CALIBRATED',len(calibration),flush=True)
    oldfile=ROOT/'cad_without_tables/ocr_results.json'
    newfile=ROOT.parent/'output/cad_fusion_refined/combined_results.json'
    old=json.loads(oldfile.read_text(encoding='utf-8'));new=json.loads(newfile.read_text(encoding='utf-8'))
    page=fitz.open(new['source'])[0]
    fields={r['id']:r for r in new['records']}
    fields.update({r['id']:r for r in new['frame_labels']})
    records=[];cache={}
    for method,data in [('original',old),('refined',new)]:
        for index,g in enumerate(data['detections']):
            if len(g['text'])!=1 or g['text'] not in exemplars or g.get('engine')=='native_pdf_text':continue
            b=g['box_pdf'];cap=max(1.,b[3]-b[1])
            if method=='original':
                key=(method,g['proposal']);proposal=old['proposals'][g['proposal']]
                members=[r for r in old['detections'] if r.get('proposal')==g['proposal']]
                boxes=[proposal['box']]+[r['box_pdf'] for r in members]
                fb=[min(b[k] for b in boxes) for k in (0,1)]+[max(b[k] for b in boxes) for k in (2,3)]
                sizes=[r['box_pdf'][3]-r['box_pdf'][1] for r in members if len(r['text'])==1 and r['text'].isalnum()]
                fieldcap=float(np.median(sizes)) if sizes else cap
            else:
                key=(method,g['field_id']);field=fields[g['field_id']]
                fb=field['box_pdf'];fieldcap=field['cap']
            if key not in cache:
                pad=fieldcap*.3
                image,origin=render(page,[fb[0]-pad,fb[1]-pad,fb[2]+pad,fb[3]+pad],96/fieldcap)
                cache[key]=(image,origin,96/fieldcap)
            image,origin,scale=cache[key]
            pad=cap*.035
            left,top=[max(0,int(np.floor((b[k]-pad-origin[k])*scale))) for k in range(2)]
            right,bottom=[min(image.shape[1 if k==0 else 0],int(np.ceil((b[k+2]+pad-origin[k])*scale))) for k in range(2)]
            crop=image[top:bottom,left:right]
            if crop.size:
                crop=cv2.copyMakeBorder(crop,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
                result=reader.predict(crop,proposed=g['text'])
            else:result=dict(text='',distance=100.,margin=0.,proposed_distance=100.,alternatives=[])
            mismatch=result['proposed_distance']>limits[g['text']]
            review=mismatch and result['text']!=g['text'] and result['margin']>.8
            records.append(dict(id=f'{method}_{index}',method=method,index=index,
                text=g['text'],score=g['score'],box_pdf=b,stroke=result,
                crop_valid=bool(crop.size),class_limit=limits[g['text']],threshold_review=mismatch,conflict_review=review))
            if len(records)%100==0:
                print('STROKE PROPOSALS',len(records),flush=True)
                (OUT/'checkpoint.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
    result=dict(inference_llm=False,answers_loaded=False,training=False,scope='62 supplied glyph exemplars; local directional stroke parts + symmetric spatial distances. Review flags only, no deleted or corrected text.',
                thresholds='Per-class 99th percentile of 8 exemplar distortions plus 1 normalized pixel. Conflict review also requires a different nearest exemplar and margin >0.8 pixel.',
                source_sha256=new['source_sha256'],baseline_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [oldfile,newfile]},records=records)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STROKE TEMPLATE COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
