"""Freeze raw/local inference before reading evaluation labels; no CAD edits.

Compare old 67, new 68, old dual-reader and guarded bar supplementation.
"""
import argparse
import hashlib
from collections import Counter
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from evaluate_preserved_reader import match
from evaluate_real_validation import alignment
from infer_real_validation import render, inside, pdf_boxes, as_lines
from prepare import read_image, write_image
from preserved_reader import merge_predictions
from train import overlap, read_labels, read_rows
from train_vertical_bar import ROOT, OLD, NAMES
from vertical_bar_reader import VerticalBarReader

OUT = ROOT.parent/'output/cad_vertical_bar_audit'
REAL = ROOT/'real_font_validation'
BAR_WEIGHTS=None
ADAPT_LARGE=False


def boxes(result):
    return result.boxes.cpu().numpy().data.tolist()


def metrics(total):
    tp,fp,fn = map(int,total)
    return dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))


def failure_reason(target, predicted, low):
    cid, rect = target
    same = sorted([(overlap(np.asarray(b[:4]),rect),b) for b in predicted if int(b[5])==cid],key=lambda p:-p[0])
    wrong = sorted([(overlap(np.asarray(b[:4]),rect),b) for b in predicted if int(b[5])!=cid],key=lambda p:-p[0])
    low_same = sorted([(overlap(np.asarray(b[:4]),rect),b) for b in low if int(b[5])==cid],key=lambda p:-p[0])
    if wrong and wrong[0][0] >= .5:
        reason='wrong_class'
    elif low_same and low_same[0][0] >= .5 and low_same[0][1][4] < .1:
        reason='below_confidence'
    elif same and same[0][0] >= .1:
        reason='box_localization'
    else:
        reason='missing_or_suppressed'
    return dict(reason=reason,best_same=same[:1],best_other=wrong[:1],best_low_same=low_same[:1])


def inference():
    path=OUT/'frozen_predictions.json'
    if path.exists():
        payload=json.loads(path.read_text(encoding='utf-8'))
        for name,file in payload['weights'].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==payload['model_sha256'][name]
        return payload
    reader=VerticalBarReader(BAR_WEIGHTS,adapt_large=ADAPT_LARGE)
    raw_old=YOLO(str(reader.paths['symbols']))
    raw_new=YOLO(str(reader.paths['vertical_bar']))
    weights=dict(old=str(reader.paths['symbols']),new=str(reader.paths['vertical_bar']),letters=str(reader.paths['letters']))
    jobs=[]
    for group,directory,prefix in [('old_alphanumeric',ROOT/'alphanumeric',''),('old_punctuation',OLD,''),('new_bar',ROOT/'vertical_bar','bar_')]:
        for file in sorted((directory/'images/test').glob(prefix+'*.png')):
            jobs.append(dict(id=group+'_'+file.stem,group=group,path=str(file),label_path=str(directory/'labels/test'/(file.stem+'.txt'))))
    manifest=json.loads((REAL/'regions.json').read_text(encoding='utf-8'))
    source=ROOT/'cad_without_tables/source.pdf'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_sha256']
    page=fitz.open(source)[0]
    for region in manifest['regions']:
        if region['kind']=='korean':
            continue
        cap=region['reference_cap_height_pdf']; b=region['box_pdf']
        for height in (32,40,56):
            scale=height/cap
            im,origin=render(page,[b[0]-.2*cap,b[1]-.2*cap,b[2]+.2*cap,b[3]+.2*cap],scale)
            file=OUT/'real_inputs'/f'{region["id"]}_h{height}.png'
            write_image(file,im)
            jobs.append(dict(id=f'real_{region["id"]}_h{height}',group='real',path=str(file),region=region,scale=scale,origin=origin))
    for start in range(0,len(jobs),8):
        batch=jobs[start:start+8];images=[read_image(Path(r['path'])) for r in batch]
        for key,model in [('raw_old',raw_old),('raw_new',raw_new)]:
            pred=model.predict(images,imgsz=384,conf=.1,iou=.5,end2end=False,agnostic_nms=False,device=0,batch=8,verbose=False)
            for job,p in zip(batch,pred):job[key]=boxes(p)
        for job,im in zip(batch,images):
            result,baseline,detail=reader.predict(im,job['path'],'tiles' if job['group']=='real' else 'full',384)
            original,_=merge_predictions(boxes(baseline),detail['symbols'],detail['cleaned'])
            job['dual_old']=original.tolist();job['guarded']=boxes(result)
            job['bar_decisions']=detail['vertical_bar']
            job['shape']=list(im.shape[:2])
        print('AUDIT INFERENCE',min(start+8,len(jobs)),len(jobs),flush=True)
    payload=dict(answers_loaded=False,llm_inference=False,word_correction=False,weights=weights,
        model_sha256={k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in weights.items()},
        source_sha256=manifest['source_sha256'],confidence=.1,iou=.5,records=jobs)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


def render_failure(record, target, destination):
    im=read_image(Path(record['path']));h,w=im.shape[:2]
    rect=target[1];pad=max(8,(rect[3]-rect[1])*.5)
    x0=max(0,int(rect[0]-pad));x1=min(w,int(rect[2]+pad))
    y0=max(0,int(rect[1]-pad));y1=min(h,int(rect[3]+pad))
    panels=[]
    for key in ['raw_old','raw_new']:
        part=im[y0:y1,x0:x1].copy()
        chosen=[]
        for b in record[key]:
            if overlap(np.asarray(b[:4]),rect)>.05:
                chosen.append([b[0]-x0,b[1]-y0,b[2]-x0,b[3]-y0,*b[4:]])
        r=Results(orig_img=part,path=key,names=dict(enumerate(NAMES)),boxes=np.asarray(chosen,dtype=np.float32).reshape(-1,6))
        panels.append(cv2.resize(r.plot(line_width=1,font_size=9),(240,200)))
    write_image(destination,np.hstack(panels))


def evaluate(payload):
    summary={};lost=[];gained=[];class_counts={};real_rows=[]
    raw_new=YOLO(payload['weights']['new'])
    references={r['id']:r for r in json.loads((REAL/'answers.json').read_text(encoding='utf-8'))['answers']}
    for rec in payload['records']:
        if rec['group']=='real':
            target=rec['region'];reference=references[target['id']]
            values={}
            for version in ['raw_old','raw_new','dual_old','guarded']:
                # Raw full-image model boxes and native-tile boxes both use input pixels.
                selected=[]
                for b in rec[version]:
                    p=[b[0]/rec['scale']+rec['origin'][0],b[1]/rec['scale']+rec['origin'][1],b[2]/rec['scale']+rec['origin'][0],b[3]/rec['scale']+rec['origin'][1],*b[4:]]
                    if inside(p,target['box_pdf'],target['reference_cap_height_pdf']*.035):selected.append(p)
                text='\n'.join(as_lines(selected,dict(enumerate(NAMES))))
                values[version]=dict(text=text,score=alignment(reference['reference'],text,reference.get('ambiguities')),detections=len(selected))
            real_rows.append(dict(id=rec['id'],region_id=target['id'],kind=target['kind'],reference=reference['reference'],values=values,accepted_bars=rec['bar_decisions']['accepted']))
            continue
        group=rec['group'];truth=read_labels(Path(rec['label_path']),rec['shape'][1],rec['shape'][0])
        totals=summary.setdefault(group,{k:np.zeros(3,dtype=int) for k in ['raw_old','raw_new','dual_old','guarded']})
        matched={}
        for version in totals:
            hit,fp,fn=match(rec[version],truth);totals[version]+=[len(hit),fp,fn];matched[version]=hit
            classes=class_counts.setdefault(group,{}).setdefault(version,{})
            for char in ['I','l','i','1','|']:
                cid=NAMES.index(char)
                hit_c,fp_c,fn_c=match([b for b in rec[version] if int(b[5])==cid],[t for t in truth if t[0]==cid])
                classes.setdefault(char,np.zeros(3,dtype=int));classes[char]+=[len(hit_c),fp_c,fn_c]
        missing=matched['raw_old']-matched['raw_new']
        if missing:
            low=boxes(raw_new.predict(read_image(Path(rec['path'])),imgsz=384,conf=.001,iou=.5,end2end=False,agnostic_nms=False,device=0,verbose=False,max_det=1000)[0])
            for j in sorted(missing):
                cid,rect=truth[j]
                detail=failure_reason(truth[j],rec['raw_new'],low)
                before=max([(overlap(np.asarray(b[:4]),rect),b) for b in rec['raw_old'] if int(b[5])==cid],key=lambda p:p[0])
                image=f'failures/{rec["id"]}_glyph{j}.png'
                render_failure(rec,truth[j],OUT/image)
                lost.append(dict(id=rec['id'],truth_index=j,label=NAMES[cid],box=rect.tolist(),before=before,image=image,**detail))
        for j in sorted(matched['raw_new']-matched['raw_old']):
            gained.append(dict(id=rec['id'],label=NAMES[truth[j][0]],truth_index=j))
        rec['guarded_previously_correct_old_lost']=[j for j in matched['dual_old']-matched['guarded'] if truth[j][0]<67]
    summary={g:{v:metrics(t) for v,t in row.items()} for g,row in summary.items()}
    class_counts={g:{v:{c:metrics(t) for c,t in row.items()} for v,row in versions.items()} for g,versions in class_counts.items()}
    real_summary={}
    for key in ['raw_old','raw_new','dual_old','guarded']:
        positives=[r for r in real_rows if r['kind']!='drawing_negative']
        n=sum(r['values'][key]['score']['reference_characters'] for r in positives)
        errors=sum(r['values'][key]['score']['edit_distance'] for r in positives)
        real_summary[key]=dict(views=len(positives),unique_positive_regions=len(set(r['region_id'] for r in positives)),exact_views=sum(r['values'][key]['score']['exact'] for r in positives),characters=n,errors=errors,cer=errors/max(n,1),negative_false_detections=sum(r['values'][key]['detections'] for r in real_rows if r['kind']=='drawing_negative'))
    old_losses=[r for r in lost if not r['id'].startswith('new_bar_')]
    real_changes=[r for r in real_rows if r['values']['dual_old']['text'] != r['values']['guarded']['text']]
    result=dict(scope='Retrospective same-template synthetic tests and 22 fixed real regions at 3 scales, not independent deployment accuracy. References loaded only after frozen inference.',
        frozen_predictions_sha256=hashlib.sha256((OUT/'frozen_predictions.json').read_bytes()).hexdigest(),summary=summary,selected_classes=class_counts,
        lost=lost,gained=gained,old_set_loss_reasons=dict(Counter(r['reason'] for r in old_losses)),old_set_loss_labels=dict(Counter(r['label'] for r in old_losses)),
        guarded_correct_old_lost=[dict(id=r['id'],truth_indices=r['guarded_previously_correct_old_lost']) for r in payload['records'] if r.get('guarded_previously_correct_old_lost')],
        real_summary=real_summary,real_rows=real_rows,real_guarded_changes=real_changes)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(result)
    print(json.dumps({k:result[k] for k in ['summary','old_set_loss_reasons','old_set_loss_labels','guarded_correct_old_lost','real_summary','real_guarded_changes']},ensure_ascii=True),flush=True)
    return result


def write_report(result):
    rows=[]
    for dataset,versions in result['summary'].items():
        for version,s in versions.items():
            rows.append(f'<tr><td>{dataset}</td><td>{version}</td><td>{s["tp"]}</td><td>{s["fp"]}</td><td>{s["fn"]}</td><td>{s["precision"]:.2%}</td><td>{s["recall"]:.2%}</td></tr>')
    cases=''.join('<article><h3>'+escape(r['id']+' / '+r['label']+' / '+r['reason'])+'</h3><img src="'+r['image']+'"><pre>'+escape(json.dumps({k:r[k] for k in ['before','best_same','best_other','best_low_same']},indent=2))+'</pre></article>' for r in result['lost'])
    page='''<!doctype html><meta charset="utf-8"><title>| 추가학습 성능 재검사</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}article{border-top:1px solid #aaa}pre{white-space:pre-wrap}img{max-width:100%}</style>
<h1>| 추가학습 성능 재검사</h1><p>raw_old: 기존 67클래스 모델, raw_new: 추가학습 68클래스 모델, dual_old: 기존 62+67 결합, guarded: 기존 결합에 |만 보충.</p><p>합성 검사는 학습과 같은 글꼴 템플릿입니다. 실제 도면은 기존에 검사했던 22개 영역의 3가지 크기이며, 66개 독립 영역이 아닙니다. 추론을 저장한 뒤 정답지를 읽어 평가했습니다.</p>
<table><tr><th>세트</th><th>방식</th><th>정답 검출</th><th>오탐</th><th>누락</th><th>정밀도</th><th>재현율</th></tr>'''+''.join(rows)+'''</table><h2>실제 도면 검사</h2><pre>'''+escape(json.dumps(result['real_summary'],indent=2))+'''</pre><h2>기존에 맞았으나 새로 틀린 사례</h2><p>이미지 왼쪽은 기존 모델, 오른쪽은 추가학습 모델입니다. wrong_class=다른 문자로 변경, below_confidence=정답 후보 점수 하락, box_localization=박스 위치 차이, missing_or_suppressed=해당 검사에서 후보 확인 불가.</p>'''+cases+'''<p><a href="analysis.json">상세 수치</a> · <a href="frozen_predictions.json">수정하지 않은 모델 출력</a></p>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')


def main():
    global OUT,BAR_WEIGHTS,ADAPT_LARGE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bar-weights',type=Path)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--adapt-large',action='store_true')
    args=parser.parse_args();OUT=args.output.resolve();BAR_WEIGHTS=args.bar_weights;ADAPT_LARGE=args.adapt_large
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    evaluate(inference())


if __name__=='__main__':main()
