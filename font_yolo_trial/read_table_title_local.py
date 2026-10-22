"""Recognize the selected table title using local models, without answer strings."""
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import pymupdf as fitz
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from models_local import _offline
from paddle_ocr_local import model_dirs
from preserved_reader import PreservedReader
from iterate_fusion import vector_fields,word_regions,fuse_words,lines,area_coverage
from ctc_korean_fusion import vector_line_image,attach_confidence,choose_spans
from vector_extra_symbols import supplement
from infer_real_validation import render

OUT=ROOT/'output/cad_table_title_local'


def spans(raw,image_width,origin,scale,field):
    value=raw.get('rec_text')
    if not isinstance(value,(list,tuple)) or len(value)!=2:return []
    text,info=value
    if len(info)<5:return []
    columns=float(info[0]);positions=[x for group in info[2] for x in group];scores=info[4]
    if len(text)!=len(positions) or len(text)!=len(scores) or columns<=0:return []
    box=field['box'];cap=field['cap'];result=[]
    for match in re.finditer(r'[0-9]*[\uac00-\ud7a3]+',text):
        token=match.group();a,b=match.span();score=float(np.mean(scores[a:b]))
        if score<(.97 if len(token)==1 else .88):continue
        left=max(box[0],origin[0]+positions[a]/columns*image_width/scale-cap*.58)
        right=min(box[2],origin[0]+positions[b-1]/columns*image_width/scale+cap*.58)
        if right<=left or len(token)>max(2,(right-left)/cap*2.2):continue
        result.append(dict(text=token,accepted=token,score=score,box_pdf=[left,box[1],right,box[3]],engine='local_title_ctc_span'))
    return result


def recognize_field(page,field,out,reader,model):
    out.mkdir(parents=True,exist_ok=True)
    record=dict(id=field['id'],box_pdf=field['box'],cap=field['cap'])
    jobs=[]
    for height,pad in [(40,.15),(56,.25),(80,.35)]:
        image,origin,scale=vector_line_image(field,height,pad)
        jobs.append((f'vector_h{height}',image,origin,scale))
        ink=cv2.erode(image,np.ones((2,2),np.uint8))
        jobs.append((f'ink_h{height}',ink,origin,scale))
        cap=field['cap'];box=field['box'];clip=[box[0]-pad*cap,box[1]-pad*cap,box[2]+pad*cap,box[3]+pad*cap]
        native,origin=render(page,clip,height/cap)
        jobs.append((f'native_h{height}',native,origin,height/cap))
    glyph_runs=vector_fields(page,[record],reader,heights=(40,56,80),candidates=[field])
    (out/'yolo_predictions.json').write_text(json.dumps(dict(llm_inference=False,answers_loaded=False,results=glyph_runs),ensure_ascii=False,indent=2),encoding='utf-8')
    diagnostics=[];proposals=[]
    for variant,image,origin,scale in jobs:
        cv2.imencode('.png',image)[1].tofile(out/(variant+'.png'))
        prediction=next(iter(model.predict(input=image,batch_size=1,return_word_box=True)))
        raw=prediction.json.get('res',prediction.json)
        found=spans(raw,image.shape[1],origin,scale,field)
        proposals.extend({**w,'variant':variant} for w in found)
        diagnostics.append(dict(variant=variant,raw=raw,spans=found,origin=origin,scale=scale,image_width=image.shape[1]))
        print('FIELD_OCR',field['id'],variant,json.dumps(raw.get('rec_text'),ensure_ascii=True)[:180],flush=True)
    # Single syllables need repeated spatial support; no expected Korean words.
    proposals=[p for p in proposals if len(p['text'])>1 or len({q['variant'] for q in proposals if q['text']==p['text'] and max(area_coverage(p['box_pdf'],q['box_pdf']),area_coverage(q['box_pdf'],p['box_pdf']))>.5})>=3]
    selected=choose_spans(proposals,field['cap'])
    combined=[]
    for run in glyph_runs:
        glyphs=supplement(run['glyphs'],field,selected)
        fused,suppressed=fuse_words(glyphs,selected)
        combined.append(dict(config=run['config'],yolo_lines=lines(glyphs),fused_lines=lines(fused),glyphs=glyphs,fused_glyphs=fused,suppressed=suppressed))
    report=dict(llm_inference=False,answers_loaded=False,word_correction=False,source_geometry='source_geometry.json',
        field=field,ocr_diagnostics=diagnostics,korean_spans=selected,results=combined)
    (out/'recognition.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TITLE_RESULT',json.dumps([{k:r[k] for k in ['config','yolo_lines','fused_lines']} for r in combined],ensure_ascii=True),flush=True)
    return report


def main():
    torch.set_num_threads(4)
    source=json.loads((OUT/'source_geometry.json').read_text(encoding='utf-8'))
    page=fitz.open(source['source_pdf'])[0]
    with _offline(paddle_cache=ROOT/'.model_cache/paddlex'):
        reader=PreservedReader()
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        attach_confidence(model)
        recognize_field(page,source['field'],OUT,reader,model)


if __name__=='__main__':main()
