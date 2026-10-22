"""Spatial Korean/Latin fusion using local CTC positions, never expected words."""
import bisect
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from infer_real_validation import render
from iterate_fusion import ROOT, accepted_korean, area_coverage, fuse_words, lines
from vector_font_probe import load_candidates

BASE=ROOT.parent/'output/cad_fusion_iterations'
OUT=ROOT.parent/'output/cad_fusion_refined'


def ctc_confidences(probabilities):
    indices=probabilities.argmax(-1);confidence=probabilities.max(-1)
    result=[]
    for ids,scores in zip(indices,confidence):
        selected=np.r_[True,ids[1:]!=ids[:-1]] & (ids!=0)
        result.append(scores[selected].tolist())
    return result


def attach_confidence(model):
    # Read extra scores from the installed CTC postprocessor, without changing its text.
    predictor=model.paddlex_predictor;original=predictor.post_op
    def postprocess(pred,**kwargs):
        texts,scores=original(pred,**kwargs)
        confidence=ctc_confidences(np.asarray(pred[0]))
        for i,value in enumerate(texts):
            if isinstance(value,tuple):
                text,positions=value
                if len(text)!=len(confidence[i]):raise ValueError('CTC character/score count mismatch')
                texts[i]=(text,[*positions,confidence[i]])
        return texts,scores
    predictor.post_op=postprocess


def korean_spans(raw,image_width,origin,scale,cap,field_box):
    value=raw.get('rec_text')
    if not isinstance(value,(list,tuple)) or len(value)!=2:return []
    text,info=value
    if len(info)<5:return []
    columns=float(info[0]);positions=[x for group in info[2] for x in group];scores=info[4]
    if len(text)!=len(positions) or len(text)!=len(scores) or columns<=0:return []
    spans=[]
    for match in re.finditer(r'[0-9]*[가-힣]+',text):
        token=match.group();a,b=match.span();confidence=float(np.mean(scores[a:b]))
        if len(token)<2 or confidence<.88:continue
        left=origin[0]+positions[a]/columns*image_width/scale-cap*.58
        right=origin[0]+positions[b-1]/columns*image_width/scale+cap*.58
        left=max(field_box[0],left);right=min(field_box[2],right)
        if right<=left or len(token)>max(2,(right-left)/cap*2.2):continue
        spans.append(dict(text=token,accepted=token,score=confidence,
            box_pdf=[left,field_box[1],right,field_box[3]],engine='ctc_korean_span'))
    return spans


def choose_spans(candidates,cap):
    # Votes are observations from different input variants, not model confidence calibration.
    anchors=[w for w in candidates if w.get('variant')=='previous_word']
    filtered=[]
    for item in candidates:
        token=''.join(item['accepted'].split())
        conflicts=[]
        for anchor in anchors:
            old=''.join(anchor['accepted'].split())
            overlaps=max(area_coverage(item['box_pdf'],anchor['box_pdf']),area_coverage(anchor['box_pdf'],item['box_pdf']))>.5
            if overlaps and token not in old and old not in token:conflicts.append(anchor)
        if conflicts:
            evidence={w.get('variant') for w in candidates if w.get('variant','').startswith('vector_') and
                      ''.join(w['accepted'].split())==token and w['score']>=.98 and
                      max(area_coverage(w['box_pdf'],item['box_pdf']),area_coverage(item['box_pdf'],w['box_pdf']))>.5}
            if len(evidence)<3:continue
            anchor=max(conflicts,key=lambda w:w['score'])
            if len(token)==len(''.join(anchor['accepted'].split())):item={**item,'box_pdf':anchor['box_pdf']}
        filtered.append(item)
    unique=[]
    for item in sorted(filtered,key=lambda w:-w['score']):
        old=next((w for w in unique if w['accepted']==item['accepted'] and
                  (area_coverage(w['box_pdf'],item['box_pdf'])>.5 or area_coverage(item['box_pdf'],w['box_pdf'])>.5)),None)
        if old:
            old['variants']=sorted(set(old['variants']+[item.get('variant','baseline')]))
            if item.get('variant')=='previous_word':old['box_pdf']=item['box_pdf']
        else:unique.append({**item,'variants':[item.get('variant','baseline')]})
    ordered=sorted(unique,key=lambda w:w['box_pdf'][2]);ends=[w['box_pdf'][2] for w in ordered]
    best=[(0.,[])]
    for i,item in enumerate(ordered):
        previous=bisect.bisect_right(ends,item['box_pdf'][0]+cap*.12,0,i)
        count=len(re.sub(r'\s','',item['accepted']))
        weight=count*(item['score']-.70)+min(3,len(item['variants']))*.008
        take=(best[previous][0]+weight,best[previous][1]+[item])
        best.append(take if take[0]>best[-1][0] else best[-1])
    return best[-1][1]


def vector_line_image(path,height,pad):
    box=path['box'];cap=path['cap'];scale=height/cap;origin=[box[0]-pad*cap,box[1]-pad*cap]
    width=int(np.ceil((box[2]-box[0]+2*pad*cap)*scale))+1
    rows=int(np.ceil((box[3]-box[1]+2*pad*cap)*scale))+1
    high=np.full((rows*3,width*3,3),255,np.uint8)
    for group in path['groups']:
        for segment in group['segments']:
            points=[]
            for offset in [0,2]:
                # Relative quantization removes absolute-page subpixel changes, not stroke shape.
                normalized=[round((segment[offset+i]-box[i])/cap/.005)*.005 for i in range(2)]
                points.append(tuple(round((v+pad)*height*3) for v in normalized))
            cv2.line(high,points[0],points[1],(0,0,0),max(1,round(path['width']*scale*3)),cv2.LINE_AA)
    return cv2.resize(high,(width,rows),interpolation=cv2.INTER_AREA),origin,scale


def recognize_spans(page,data,vector_only=False):
    sys.path.insert(0,str(ROOT.parent))
    from models_local import _offline
    from paddle_ocr_local import model_dirs
    jobs=[];proposals={r['id']:[] for r in data['records']};diagnostics=[]
    paths={f"P{p['path_id']}":p for p in load_candidates(page)} if vector_only else {}
    for record in data['records']:
        identity=record['id'];cap=record['cap'];box=record['box_pdf']
        for word in data['words'][identity]:
            if word.get('accepted'):proposals[identity].append({**word,'variant':'previous_word'})
        if vector_only:
            for height,pad in [(40,.15),(56,.25),(80,.35)]:
                image,origin,scale=vector_line_image(paths[identity],height,pad)
                jobs.append((record,image,origin,scale,f'vector_h{height}_p{pad}'))
            continue
        for pad in [.10,.25,.40]:
            clip=[box[0]-pad*cap,box[1]-pad*cap,box[2]+pad*cap,box[3]+pad*cap]
            scale=56/cap;image,origin=render(page,clip,scale)
            jobs.append((record,image,origin,scale,f'line_pad{pad}'))
        # Retry word crops containing Hangul; no lexical assumptions about their content.
        for index,word in enumerate(data['words'][identity]):
            if not re.search('[가-힣]',word['text']) or word['score']>=.995:continue
            wb=word['box_pdf']
            for pad in [.10,.30]:
                scale=56/cap
                image,origin=render(page,[wb[0]-pad*cap,wb[1]-pad*cap,wb[2]+pad*cap,wb[3]+pad*cap],scale)
                jobs.append((record,image,origin,scale,f'word{index}_pad{pad}'))
    with _offline(paddle_cache=ROOT.parent/'.model_cache/paddlex'):
        from paddleocr import TextRecognition
        _,directory=model_dirs()
        model=TextRecognition(model_name='korean_PP-OCRv5_mobile_rec',model_dir=str(directory),device='cpu',enable_mkldnn=False,cpu_threads=4)
        attach_confidence(model)
        # Batch size 1 keeps feature-column / input-width alignment independent of batch sorting.
        for index,(record,image,origin,scale,variant) in enumerate(jobs):
            result=next(iter(model.predict(input=image,batch_size=1,return_word_box=True)))
            raw=result.json.get('res',result.json)
            candidates=korean_spans(raw,image.shape[1],origin,scale,record['cap'],record['box_pdf'])
            for word in candidates:proposals[record['id']].append({**word,'variant':variant})
            diagnostics.append(dict(id=record['id'],variant=variant,raw=raw,spans=candidates))
            if index%20==0:
                print('CTC',index,'/',len(jobs),flush=True)
                (OUT/'ctc_checkpoint.json').write_text(json.dumps(diagnostics,ensure_ascii=False),encoding='utf-8')
    selected={r['id']:choose_spans(proposals[r['id']],r['cap']) for r in data['records']}
    return selected,diagnostics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');parser.add_argument('--vectors',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    source_file=BASE/'automatic_results.json';data=json.loads(source_file.read_text(encoding='utf-8'))
    page=fitz.open(data['source'])[0]
    if args.reuse or args.vectors:
        diagnostics=json.loads((OUT/'ctc_diagnostics.json').read_text(encoding='utf-8'))
        if args.vectors and not any(d['variant'].startswith('vector_') for d in diagnostics):
            _,extra=recognize_spans(page,data,vector_only=True);diagnostics.extend(extra)
        proposals={r['id']:[{**w,'variant':'previous_word'} for w in data['words'][r['id']] if w.get('accepted')] for r in data['records']}
        for record in diagnostics:
            proposals[record['id']].extend({**w,'variant':record['variant']} for w in record['spans'])
        selected={r['id']:choose_spans(proposals[r['id']],r['cap']) for r in data['records']}
    else:selected,diagnostics=recognize_spans(page,data)
    records=[];detections=[]
    for old in data['records']:
        glyphs=list(old['glyphs'])
        # A colon in this source font has taller dot separation than a partial Korean component.
        glyphs=[g for g in glyphs if not(g['engine']=='vector_geometry_symbol' and g['text']==':' and g['box_pdf'][3]-g['box_pdf'][1]<old['cap']*.45)]
        fused,suppressed=fuse_words(glyphs,selected[old['id']])
        records.append({**old,'glyphs':glyphs,'fused_glyphs':fused,'suppressed':suppressed,'fused_lines':lines(fused)})
        detections.extend({**g,'field_id':old['id']} for g in fused)
    detections.extend(g for g in data['detections'] if g['engine']=='native_pdf_text')
    output={**data,'records':records,'detections':detections,'words':selected,
        'method':'CTC character positions and per-character probabilities; Korean spans only; spatial variant selection. English strings remain YOLO outputs.',
        'baseline_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
        'inference_llm':False,'answers_loaded':False,'word_correction':False,'training':False}
    (OUT/'refined_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'ctc_diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CTC REFINEMENT COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
