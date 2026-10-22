"""Compare independent cropped readers. No answers, language labels or LLM in inference."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import torch
from ultralytics import YOLO

from infer_real_validation import render, as_lines, inside
from preserved_reader import PreservedReader, foreground
from prepare import write_image
from prepare_real_validation import cap_band
from vector_font_probe import load_candidates, glyph_canvas, geometric_symbol

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'fusion_iterations'


def field_box(region):
    a=region['box_pdf'];b=region.get('source_box_before_height_measurement',a)
    return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]


def word_regions(image,cap):
    ink=foreground(image)>0
    columns=ink.any(0)
    spans=np.flatnonzero(np.diff(np.r_[False,columns,False].astype(np.int8))).reshape(-1,2)
    groups=[]
    for left,right in spans:
        if groups and left-groups[-1][1]<=cap*.55:groups[-1][1]=int(right)
        else:groups.append([int(left),int(right)])
    regions=[]
    for left,right in groups:
        ys,_=np.where(ink[:,left:right])
        if len(ys):regions.append([left,int(ys.min()),right,int(ys.max())+1])
    return regions


def accepted_korean(text,score):
    text=text.strip()
    if not re.search('[가-힣]',text):return None
    if re.search('[A-Za-z]',text):return None
    if not re.fullmatch(r"[가-힣ㄱ-ㅎㅏ-ㅣ0-9\s.:/()'\-]+",text):return None
    # Mixed digit/Hangul words such as 2회 require a stronger score.
    minimum=.90 if re.search('[0-9]',text) else .80
    return text if score>=minimum else None


def area_coverage(box,region):
    intersection=max(0,min(box[2],region[2])-max(box[0],region[0]))*max(0,min(box[3],region[3])-max(box[1],region[1]))
    return intersection/max(1e-8,(box[2]-box[0])*(box[3]-box[1]))


def fuse_words(glyphs,words):
    accepted=[w for w in words if w.get('accepted')]
    kept=[];suppressed=[]
    for glyph in glyphs:
        box=glyph['box_pdf'];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        covering=None
        for word in accepted:
            wb=word['box_pdf'];pad=(wb[3]-wb[1])*.08
            expanded=[wb[0]-pad,wb[1]-pad,wb[2]+pad,wb[3]+pad]
            if not inside(box,expanded) or area_coverage(box,expanded)<.6:continue
            # OCR often omits trailing punctuation; retain the independently detected mark.
            if glyph['text'] in ":-/.'_" and glyph['text'] not in word['accepted']:
                edge=min(abs(cx-wb[0]),abs(cx-wb[2]))
                if edge<=(wb[3]-wb[1])*.3:continue
            covering=word;break
        if covering:suppressed.append(glyph)
        else:kept.append(glyph)
    return kept+[dict(text=w['accepted'],score=w['score'],box_pdf=w['box_pdf'],engine='korean_word_ocr') for w in accepted],suppressed


def lines(glyphs):
    # Position small punctuation on the nearest text row, not a separate baseline row.
    if not glyphs:return []
    names={};boxes=[]
    tall=[g['box_pdf'][3]-g['box_pdf'][1] for g in glyphs if len(g['text'])==1 and g['text'].isalnum()]
    cap=float(np.median(tall)) if tall else 10.
    normal=[g for g in glyphs if g['box_pdf'][3]-g['box_pdf'][1]>=cap*.5]
    for glyph in glyphs:
        text=glyph['text'];index=next((k for k,v in names.items() if v==text),None)
        if index is None:index=len(names);names[index]=text
        box=list(glyph['box_pdf'])
        if box[3]-box[1]<cap*.45 and normal:
            near=min(normal,key=lambda g:abs((g['box_pdf'][1]+g['box_pdf'][3])/2-(box[1]+box[3])/2))
            nb=near['box_pdf'];box[1],box[3]=nb[1],nb[3]
        if box[2]<=box[0]:box[2]=box[0]+.01
        boxes.append([*box,glyph['score'],index])
    return as_lines(boxes,names)


def owned_by_field(glyph,field):
    box=glyph['box_pdf'];target=field['box_pdf']
    fraction=max(0,min(box[2],target[2])-max(box[0],target[0]))/max(.01,box[2]-box[0])
    short=box[3]-box[1]<field['cap']*.25
    return inside(box,target,field['cap']*.10) and (not short or fraction>=.90)


def vector_symbol(group,cap,baseline):
    symbol=geometric_symbol(group,cap,baseline)
    if symbol:return symbol
    a,b,c,d=group['box'];width=c-a;height=d-b
    if width>cap*.35 or height>cap*.85:return None
    if height<=cap*.22 and width<=cap*.25 and b>=baseline-cap*.20:return '.'
    if not (cap*.35<=height<=cap*.85 and width<=cap*.25):return None
    mask=np.zeros((100,100),np.uint8);scale=64/cap
    for x0,y0,x1,y1 in group['segments']:
        cv2.line(mask,(round((x0-a)*scale+10),round((y0-b)*scale+10)),
                 (round((x1-a)*scale+10),round((y1-b)*scale+10)),255,2)
    count,_,stats,centers=cv2.connectedComponentsWithStats(mask)
    if count==3 and all(s[2]<cap*scale*.3 and s[3]<cap*scale*.3 for s in stats[1:]) and abs(centers[1][0]-centers[2][0])<cap*scale*.12:return ':'
    return None


def hybrid_glyphs(vector,raster,cap):
    kept=list(vector)
    for glyph in raster:
        if glyph['text'] not in ":-/.'_" or glyph['score']<.5:continue
        box=glyph['box_pdf']
        if any(g['text']==glyph['text'] and inside(box,[g['box_pdf'][0]-cap*.15,g['box_pdf'][1]-cap*.15,g['box_pdf'][2]+cap*.15,g['box_pdf'][3]+cap*.15]) for g in kept):continue
        tall=[g for g in kept if g['text'].isalnum() and g['box_pdf'][3]-g['box_pdf'][1]>=cap*.7]
        if any(area_coverage(box,g['box_pdf'])>.5 for g in tall):continue
        kept=[g for g in kept if not (g['text'].isalnum() and g['box_pdf'][3]-g['box_pdf'][1]<cap*.7 and area_coverage(g['box_pdf'],box)>.6)]
        kept.append({**glyph,'engine':'trained_symbol_supplement'})
    return kept


def make_fields(page,manifest):
    fields=[]
    for region in manifest['regions']:
        box=field_box(region)
        im,origin=render(page,[box[0]-1,box[1]-2,box[2]+1,box[3]+2],4)
        hint=region.get('reference_cap_height_pdf',box[3]-box[1])
        band=cap_band(im,((box[1]+box[3])/2-origin[1])*4,hint*4)
        cap=(band[1]-band[0])/4 if band else hint
        fields.append(dict(id=region['id'],box_pdf=box,cap=max(4.,cap)))
    return fields


def raster_fields(page,fields,reader,new_model=None,heights=(56,80),include_old=True):
    records=[]
    for field in fields:
        b=field['box_pdf'];cap=field['cap'];clip=[b[0]-.2*cap,b[1]-.15*cap,b[2]+.2*cap,b[3]+.15*cap]
        for height in heights:
            scale=height/cap;im,origin=render(page,clip,scale)
            if height==56 and include_old:
                result,_,_=reader.predict(im,mode='tiles')
                data=result.boxes.cpu().numpy().data.tolist()
                glyphs=[dict(text=reader.names[int(g[5])],score=g[4],box_pdf=[g[k]/scale+origin[k%2] for k in range(4)],engine='old_yolo') for g in data]
                records.append(dict(id=field['id'],config='raster_old_h56',glyphs=glyphs))
            if new_model is not None:
                from compare_punctuation import tiled
                result=tiled(new_model,im,'trial')
                data=result.boxes.cpu().numpy().data.tolist()
                glyphs=[dict(text=new_model.names[int(g[5])],score=g[4],box_pdf=[g[k]/scale+origin[k%2] for k in range(4)],engine='new_yolo') for g in data]
                records.append(dict(id=field['id'],config=f'raster_trained_h{height}',glyphs=glyphs))
        print('RASTER',field['id'],flush=True)
    return records


def vector_fields(page,fields,reader,new_model=None,heights=(40,56,80),candidates=None):
    if candidates is None:candidates=load_candidates(page)
    jobs=[];records=[]
    for field in fields:
        box=field['box_pdf'];cap=field['cap']
        groups=[];seen=[]
        for path in candidates:
            if not (path['box'][0]<box[2] and box[0]<path['box'][2] and path['box'][1]<box[3] and box[1]<path['box'][3]):continue
            for group in path['groups']:
                gb=group['box']
                if not inside(gb,box,cap*.10):continue
                if any(all(abs(x-y)<.02 for x,y in zip(gb,old)) for old in seen):continue
                seen.append(gb);groups.append((path,group))
        for height in heights:
            entry=dict(id=field['id'],config=f'vector_spaced_h{height}',glyphs=[])
            # Separate groups on white canvases. Keep every centered detection, not just one max box.
            for path,group in groups:
                symbol=vector_symbol(group,path['cap'],path['baseline'])
                if symbol:
                    entry['glyphs'].append(dict(text=symbol,score=1.,box_pdf=group['box'],engine='vector_geometry_symbol'));continue
                jobs.append((entry,path,group,glyph_canvas(group,path['cap'],height,path['width']),height))
            records.append(entry)
    for start in range(0,len(jobs),16):
        batch=jobs[start:start+16]
        results=reader.letters.predict([j[3] for j in batch],imgsz=384,conf=.1,iou=.5,agnostic_nms=True,end2end=False,device=0,batch=16,verbose=False)
        for (entry,path,group,im,height),result in zip(batch,results):
            scale=height/path['cap'];gb=group['box'];cx,cy=(gb[0]+gb[2])/2,(gb[1]+gb[3])/2
            data=result.boxes.cpu().numpy().data.tolist()
            for b in data:
                mapped=[(b[k]-192)/scale+(cx if k%2==0 else cy) for k in range(4)]
                if not inside(mapped,gb,path['cap']*.1):continue
                entry['glyphs'].append(dict(text=reader.names[int(b[5])],score=b[4],box_pdf=mapped,engine='spaced_vector_yolo'))
        if start%160==0:print('SPACED',start,'/',len(jobs),flush=True)
    return records


def ocr_words(page,fields):
    jobs=[];words={r['id']:[] for r in fields}
    for field in fields:
        b=field['box_pdf'];cap=field['cap'];scale=56/cap
        im,origin=render(page,[b[0]-.1*cap,b[1]-.1*cap,b[2]+.1*cap,b[3]+.1*cap],scale)
        for box in word_regions(im,56):
            if box[2]-box[0]<4 or box[3]-box[1]<8:continue
            crop=im[box[1]:box[3],box[0]:box[2]]
            crop=cv2.copyMakeBorder(crop,12,12,12,12,cv2.BORDER_CONSTANT,value=(255,255,255))
            word=dict(box_pdf=[box[k]/scale+origin[k%2] for k in range(4)])
            words[field['id']].append(word);jobs.append((word,crop))
    sys.path.insert(0,str(ROOT.parent))
    from models_local import _offline
    from paddle_ocr_local import model_dirs
    with _offline(paddle_cache=ROOT.parent/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        predictions=model.predict(input=[j[1] for j in jobs],batch_size=8)
        for (word,_),prediction in zip(jobs,predictions):
            raw=prediction.json.get('res',prediction.json)
            word.update(text=str(raw.get('rec_text','')),score=float(raw.get('rec_score',0)))
            word['accepted']=accepted_korean(word['text'],word['score'])
    return words


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');parser.add_argument('--trained',action='store_true');parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    path=OUT/'audit_regions.json' if args.audit else ROOT/'real_font_validation/regions.json'
    manifest=json.loads(path.read_text(encoding='utf-8'));page=fitz.open(manifest['source'])[0]
    fields=make_fields(page,manifest)
    trained_path=OUT/'spacing_training/training/weights/best.pt'
    suffix='audit' if args.audit else 'development'
    if args.reuse:
        cached=json.loads((OUT/f'{suffix}_predictions.json').read_text(encoding='utf-8'))
        assert cached['manifest_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        records=json.loads((OUT/f'{suffix}_glyph_checkpoint.json').read_text(encoding='utf-8'))
        words=cached['words'];model_hashes=cached['original_models']
    else:
        reader=PreservedReader();model=YOLO(str(trained_path)) if args.trained else None
        records=raster_fields(page,fields,reader,model)
        records.extend(vector_fields(page,fields,reader,model))
        (OUT/f'{suffix}_glyph_checkpoint.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
        words=ocr_words(page,fields)
        model_hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in reader.paths.items()}
    native=page.get_text('words')
    hybrids=[]
    for field in fields:
        vector=next((r for r in records if r['id']==field['id'] and r['config']=='vector_spaced_h40'),None)
        raster=next((r for r in records if r['id']==field['id'] and r['config']=='raster_trained_h56'),None)
        if vector is not None and raster is not None:
            hybrids.append(dict(id=field['id'],config='hybrid_spaced_h40',glyphs=hybrid_glyphs(vector['glyphs'],raster['glyphs'],field['cap'])))
    records.extend(hybrids)
    for record in records:
        field=next(f for f in fields if f['id']==record['id'])
        record['glyphs']=[g for g in record['glyphs'] if owned_by_field(g,field)]
        record['raw_lines']=lines(record['glyphs'])
        record['fused_glyphs'],record['suppressed']=fuse_words(record['glyphs'],words[record['id']])
        embedded=[dict(text=w[4],score=1.,box_pdf=list(w[:4]),engine='native_pdf_text') for w in native if inside(w[:4],field['box_pdf'])]
        if embedded:
            record['fused_glyphs']=[g for g in record['fused_glyphs'] if not any(inside(g['box_pdf'],w['box_pdf']) for w in embedded)]+embedded
        record['fused_lines']=lines(record['fused_glyphs'])
    output=dict(inference_llm=False,word_correction=False,answers_loaded=False,scope='Known-region recognition diagnostics; not automatic full-page localization accuracy.',
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=manifest['source_sha256'],
        original_models=model_hashes,
        new_model_sha256=hashlib.sha256(trained_path.read_bytes()).hexdigest() if args.trained else None,fields=fields,words=words,records=records)
    (OUT/f'{suffix}_predictions.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FUSION COMPLETE',suffix,flush=True)


if __name__=='__main__':main()
